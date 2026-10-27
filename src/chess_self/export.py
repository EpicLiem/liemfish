from __future__ import annotations

import hashlib
import json
from pathlib import Path

import chess
import numpy as np
import torch

from .model import MOVES, SOURCE_COMMIT, encode, load_model

INPUT_NAMES = ['tokens', 'self_elos', 'oppo_elos']
OUTPUT_NAMES = ['move_logits', 'timing_logits', 'timing_probabilities', 'expected_seconds']
ATOL = 2e-4
RTOL = 2e-4


class ExportRMSNorm(torch.nn.Module):
    """Decompose RMSNorm into opset-18 arithmetic without changing its weights."""

    def __init__(self, norm):
        super().__init__()
        self.weight = norm.weight
        self.eps = norm.eps if norm.eps is not None else torch.finfo(torch.float32).eps
        self.axes = tuple(range(-len(norm.normalized_shape), 0))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(self.axes, keepdim=True) + self.eps) * self.weight


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, self_elos, oppo_elos):
        moves, timing, seconds = self.model(tokens, self_elos, oppo_elos)
        return moves, timing, timing.softmax(-1), seconds


def parity_rows():
    """Real histories exercise both orientations, padding, promotions, and clocks."""
    board = chess.Board()
    history = [board.fen()]
    rows = []

    def append(name, positions, remaining=120.0):
        rows.append(dict(name=name, fen=positions[-1], history_fens=list(positions),
                         initial_seconds=180.0, increment_seconds=2.0,
                         remaining_seconds=remaining, elo_self=1500, elo_oppo=1900))

    append('white_short', history)
    for move in ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'g8f6']:
        board.push_uci(move)
        history.append(board.fen())
        if len(history) == 2:
            append('black_short_low_clock', history, .05)
        if len(history) == 8:
            append('black_full', history[-8:])
    append('white_full_low_clock', history[-8:], .01)
    promotion = chess.Board('7k/P7/8/8/8/8/7p/4K3 w - - 0 1')
    append('white_promotion', [promotion.fen()], .2)
    append('black_promotion', [promotion.mirror().fen()], .1)
    return rows


def _inputs(rows):
    encoded = [encode(row) for row in rows]
    return tuple(torch.stack([row[name] for row in encoded]).float() for name in INPUT_NAMES)


def _sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_model(model, output: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    wrapper = ExportModel(model).eval()
    rows = parity_rows()
    cases = []
    for size in (1, 2):
        for offset in range(0, len(rows), size):
            selected = rows[offset:offset + size]
            inputs = _inputs(selected)
            with torch.no_grad():
                expected = [value.numpy() for value in wrapper(*inputs)]
            actual = session.run(OUTPUT_NAMES, dict(zip(INPUT_NAMES, [x.numpy() for x in inputs])))
            errors = {}
            for name, reference, result in zip(OUTPUT_NAMES, expected, actual):
                difference = np.abs(reference - result)
                errors[name] = dict(max_abs=float(difference.max()),
                                    mean_abs=float(difference.mean()),
                                    passed=bool(np.allclose(reference, result, atol=ATOL, rtol=RTOL)))
            cases.append(dict(positions=[row['name'] for row in selected], batch_size=size,
                              outputs=errors))
    return dict(passed=all(e['passed'] for c in cases for e in c['outputs'].values()),
                atol=ATOL, rtol=RTOL, provider='CPUExecutionProvider', cases=cases)


def export_model(checkpoint: Path, output: Path) -> dict:
    """Export a complete checkpoint to FP32 ONNX, then verify CPU runtime parity.

    ``output`` is an ONNX filename; adjacent JSON files contain the ordered move
    vocabulary and a verification report. An incomplete checkpoint fails loading
    before any output is written. Callers must finish downloads before invoking.
    """
    import onnx
    import onnxruntime as ort

    checkpoint, output = Path(checkpoint), Path(output)
    if checkpoint.resolve() == output.resolve():
        raise ValueError('Checkpoint and ONNX output must be different files')
    model = load_model(checkpoint, device='cpu').float().eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    fastpath = torch.backends.mha.get_fastpath_enabled()
    norms = []
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.RMSNorm):
                norms.append((module, name, child))
                setattr(module, name, ExportRMSNorm(child))
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad():
            torch.onnx.export(
                ExportModel(model).eval(), _inputs(parity_rows()[:2]), str(output),
                input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                dynamic_axes={name: {0: 'batch'} for name in INPUT_NAMES + OUTPUT_NAMES},
                opset_version=18, dynamo=False, external_data=False,
            )
        onnx.checker.check_model(str(output))
    finally:
        for module, name, norm in norms:
            setattr(module, name, norm)
        torch.backends.mha.set_fastpath_enabled(fastpath)
    parity = verify_model(model, output)
    moves_path = output.with_suffix('.moves.json')
    moves_path.write_text(json.dumps(MOVES) + '\n')
    report = dict(format='onnx', precision='float32', opset=18,
                  checkpoint=str(checkpoint), checkpoint_sha256=_sha256(checkpoint),
                  output=str(output), output_sha256=_sha256(output), output_bytes=output.stat().st_size,
                  source_commit=SOURCE_COMMIT,
                  versions=dict(torch=torch.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__),
                  inputs=dict(tokens=['batch', 64, 99], self_elos=['batch'], oppo_elos=['batch']),
                  outputs=dict(move_logits=['batch', len(MOVES)], timing_logits=['batch', 30],
                               timing_probabilities=['batch', 30], expected_seconds=['batch']),
                  move_vocabulary=str(moves_path),
                  time_bin_centers_seconds=model.ponder_bucket_centers.tolist(), parity=parity)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    if not parity['passed']:
        raise RuntimeError(f'ONNX parity failed; see {output.with_suffix(".json")}')
    return report
