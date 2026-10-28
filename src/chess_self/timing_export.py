"""Export timing side-head tensors; this does not certify a browser runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .timing_model import ARCHITECTURE, load_timing_head

INPUT_NAMES = ["pooled", "base_logits", "side_features", "squares", "move_indices", "move_attributes", "history"]


def sample_inputs(config, batch=2):
    generator = torch.Generator().manual_seed(731 + batch)
    def rand(*shape):
        return torch.randn(shape, generator=generator)
    return (rand(batch, config.pooled_dim), rand(batch, 30), rand(batch, config.side_dim),
            rand(batch, 64, config.pooled_dim), torch.randint(0, 64, (batch, 2), generator=generator),
            rand(batch, 8), rand(batch, config.history_length, config.history_channels))


def export_timing_head(checkpoint, output):
    import onnx
    import onnxruntime as ort

    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError("Choose fresh output paths to preserve earlier exports")
    head, metadata = load_timing_head(checkpoint)
    head.float().eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(head, sample_inputs(head.config), str(output),
                          input_names=INPUT_NAMES, output_names=["timing_raw"],
                          dynamic_axes={name: {0: "batch"} for name in INPUT_NAMES + ["timing_raw"]},
                          opset_version=18, dynamo=False, external_data=False)
    onnx.checker.check_model(str(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    required = {x.name for x in session.get_inputs()}
    cases = []
    for batch in (1, 2, 5):
        inputs = sample_inputs(head.config, batch)
        with torch.no_grad():
            expected = head(*inputs).numpy()
        feeds = {name: value.numpy() for name, value in zip(INPUT_NAMES, inputs) if name in required}
        actual = session.run(["timing_raw"], feeds)[0]
        cases.append({"batch": batch, "max_absolute_error": float(np.abs(expected-actual).max()),
                      "passed": bool(np.allclose(expected, actual, atol=2e-5, rtol=2e-5))})
    report = {"architecture": ARCHITECTURE, "config": metadata["config"],
              "base_checkpoint_sha256": metadata["base_checkpoint_sha256"],
              "feature_names": metadata["feature_names"], "format": "onnx", "opset": 18,
              "inputs": {item.name: item.shape for item in session.get_inputs()},
              "output": "timing_raw", "output_semantics": "raw alpha/beta parameters; softplus+.05 before Beta construction" if head.config.output_family == "beta" else "categorical logits with versioned family edges",
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "parity": {"provider": "CPUExecutionProvider", "cases": cases,
                         "passed": all(case["passed"] for case in cases)},
              "browser_execution_verified": False,
              "scope": "Timing side head only; requires version-matched Maia representations, causal features, distribution decoder and scheduler."}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['parity']['passed']:
        raise RuntimeError("Timing-head ONNX parity failed")
    return report


BACKBONE_ARCHITECTURE = "maia3-timing-features-v1"
BACKBONE_INPUT_NAMES = ["tokens", "self_elos", "oppo_elos"]
BACKBONE_OUTPUT_NAMES = ["move_logits", "base_logits", "pooled", "squares"]


class FeatureExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        from .timing_model import MaiaFeatures
        self.extractor = MaiaFeatures(model)

    def forward(self, tokens, self_elos, oppo_elos):
        features = self.extractor(tokens, self_elos, oppo_elos)
        return tuple(features[name] for name in BACKBONE_OUTPUT_NAMES)

    def close(self):
        self.extractor.close()


def verify_maia_features(model, output):
    import onnxruntime as ort
    from .export import ATOL, RTOL, _inputs, parity_rows

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    wrapper = FeatureExportModel(model).eval()
    cases = []
    try:
        rows = parity_rows()
        with torch.no_grad():
            for batch in (1, 2):
                for start in range(0, len(rows), batch):
                    selected = rows[start:start + batch]
                    inputs = _inputs(selected)
                    expected = [v.numpy() for v in wrapper(*inputs)]
                    actual = session.run(BACKBONE_OUTPUT_NAMES, {name: value.numpy() for name, value in zip(BACKBONE_INPUT_NAMES, inputs)})
                    errors = {}
                    for name, left, right in zip(BACKBONE_OUTPUT_NAMES, expected, actual):
                        difference = np.abs(left - right)
                        errors[name] = {"max_absolute_error": float(difference.max()),
                                        "mean_absolute_error": float(difference.mean()),
                                        "passed": bool(np.allclose(left, right, atol=ATOL, rtol=RTOL))}
                    cases.append({"positions": [row['name'] for row in selected], "batch_size": len(selected), "outputs": errors})
    finally:
        wrapper.close()
    return {"provider": "CPUExecutionProvider", "atol": ATOL, "rtol": RTOL,
            "cases": cases, "passed": all(item['passed'] for case in cases for item in case['outputs'].values())}


def export_maia_features(checkpoint, output):
    """Export feature-producing Maia and verify against its original native arithmetic."""
    import onnx
    from .export import ExportRMSNorm, _inputs, parity_rows
    from .model import MOVES, SOURCE_COMMIT, load_model

    checkpoint, output = Path(checkpoint), Path(output)
    if checkpoint.resolve() == output.resolve():
        raise ValueError("Checkpoint and ONNX output must be different files")
    if any(path.exists() for path in (output, output.with_suffix('.json'), output.with_suffix('.moves.json'))):
        raise ValueError("Choose fresh output paths to preserve earlier exports")
    model = load_model(checkpoint, device='cpu').float().eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    fastpath = torch.backends.mha.get_fastpath_enabled()
    norms = []
    wrapper = None
    try:
        for module in list(model.modules()):
            for name, child in list(module.named_children()):
                if isinstance(child, torch.nn.RMSNorm):
                    norms.append((module, name, child))
                    setattr(module, name, ExportRMSNorm(child))
        torch.backends.mha.set_fastpath_enabled(False)
        # Register representation hooks after replacing export-only normalization nodes.
        wrapper = FeatureExportModel(model).eval()
        with torch.no_grad():
            torch.onnx.export(wrapper, _inputs(parity_rows()[:2]), str(output),
                              input_names=BACKBONE_INPUT_NAMES, output_names=BACKBONE_OUTPUT_NAMES,
                              dynamic_axes={name: {0: 'batch'} for name in BACKBONE_INPUT_NAMES + BACKBONE_OUTPUT_NAMES},
                              opset_version=18, dynamo=False, external_data=False)
        onnx.checker.check_model(str(output))
    finally:
        if wrapper is not None:
            wrapper.close()
        for module, name, norm in norms:
            setattr(module, name, norm)
        torch.backends.mha.set_fastpath_enabled(fastpath)
    parity = verify_maia_features(model, output)
    output.with_suffix('.moves.json').write_text(json.dumps(MOVES) + '\n')
    report = {"architecture": BACKBONE_ARCHITECTURE, "side_head_architecture": ARCHITECTURE,
              "format": "onnx", "precision": "float32", "opset": 18,
              "source_commit": SOURCE_COMMIT,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "output_bytes": output.stat().st_size,
              "inputs": {"tokens": ['batch', 64, 99], "self_elos": ['batch'], "oppo_elos": ['batch']},
              "outputs": {"move_logits": ['batch', len(MOVES)], "base_logits": ['batch', 30],
                          "pooled": ['batch', model.cfg.dim_vit], "squares": ['batch', 64, model.cfg.dim_vit]},
              "policy_logits": "Unmasked upstream vocabulary logits; apply the legal mask before move selection or surprisal.",
              "pooled_semantics": "Original last_ln applied to mean transformer square representation.",
              "squares_semantics": "Original transformer output in the encoder's current side-to-move square orientation.",
              "move_vocabulary": str(output.with_suffix('.moves.json')),
              "parity": parity, "browser_execution_verified": False,
              "scope": "Feature-producing backbone; combine with the matching timing side head, causal features, distribution decoder and scheduler."}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    if not parity['passed']:
        raise RuntimeError("Maia feature ONNX parity failed")
    return report
