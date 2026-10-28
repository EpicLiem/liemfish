import importlib.util
import json
from pathlib import Path

import pytest
import torch

from chess_self.timing_data import feature_names
from chess_self.timing_model import TimingHead, TimingHeadConfig, save_timing_head

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rebind_policy_timing.py"
spec = importlib.util.spec_from_file_location("rebind_policy_timing", SCRIPT)
rebind_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebind_module)


@pytest.fixture
def inputs(tmp_path):
    original, candidate, head, output = [tmp_path / name for name in ("old.pt", "new.pt", "head.pt", "rebound.pt")]
    weights = {name: torch.ones(2, 2) for name in rebind_module.POLICY_KEYS}
    weights["backbone.weight"] = torch.ones(2, 2)
    torch.save({"model_state_dict": weights}, original)
    weights["proj_sq_from.weight"] = torch.zeros(2, 2)
    torch.save({"model_state_dict": weights}, candidate)
    groups = ["clock", "phase", "chess", "history_window"]
    timing = TimingHead(TimingHeadConfig(side_dim=len(feature_names(groups)), pooled_dim=2, move_mode="squares"))
    save_timing_head(head, timing, base_checkpoint_sha256=rebind_module.digest(original),
                     feature_names=feature_names(groups), extra={"feature_groups": groups, "epoch": 5})
    return dict(original_checkpoint=original, checkpoint=candidate, head=head, output=output)


def test_rebind_preserves_head_weights_and_records_required_validation(inputs):
    result = rebind_module.rebind(**inputs, allow_policy_surprisal_change=True)
    before = torch.load(inputs["head"], weights_only=True)
    after = torch.load(inputs["output"], weights_only=True)
    for name, value in before["state_dict"].items():
        torch.testing.assert_close(value, after["state_dict"][name], rtol=0, atol=0)
    assert after["base_checkpoint_sha256"] == rebind_module.digest(inputs["checkpoint"])
    assert after["extra"]["epoch"] == 5
    assert result["policy_surprisal_changed"] and result["timing_validation_required"]
    assert result["original_head_sha256"] == rebind_module.digest(inputs["head"])
    with pytest.raises(FileExistsError):
        rebind_module.rebind(**inputs, allow_policy_surprisal_change=True)


def test_rebind_requires_explicit_surprisal_acknowledgement(inputs):
    with pytest.raises(ValueError, match="policy surprisal"):
        rebind_module.rebind(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("change", ["weight", "shape", "dtype", "keys", "nan"])
def test_rebind_rejects_incompatible_checkpoint(inputs, change):
    payload = torch.load(inputs["checkpoint"], weights_only=True)
    weights = payload["model_state_dict"]
    if change == "weight":
        weights["backbone.weight"] += 1
    elif change == "shape":
        weights["proj_sq_from.weight"] = torch.zeros(3, 2)
    elif change == "dtype":
        weights["proj_sq_from.weight"] = weights["proj_sq_from.weight"].double()
    elif change == "keys":
        weights["extra.weight"] = torch.zeros(2)
    else:
        weights["proj_sq_from.weight"][0, 0] = torch.nan
    torch.save(payload, inputs["checkpoint"])
    with pytest.raises(ValueError):
        rebind_module.rebind(**inputs, allow_policy_surprisal_change=True)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("change", ["binding", "groups", "names"])
def test_rebind_rejects_incompatible_head(inputs, change):
    payload = torch.load(inputs["head"], weights_only=True)
    if change == "binding":
        payload["base_checkpoint_sha256"] = "wrong"
    elif change == "groups":
        payload["extra"]["feature_groups"] = ["difficulty"]
    else:
        payload["feature_names"] = list(reversed(payload["feature_names"]))
    torch.save(payload, inputs["head"])
    with pytest.raises(ValueError):
        rebind_module.rebind(**inputs, allow_policy_surprisal_change=True)
    assert not inputs["output"].exists()


def test_comparison_scores_identical_recorded_moves(tmp_path, monkeypatch):
    from chess_self.model import PersonalModel
    from test_model import row, tiny

    script = SCRIPT.with_name("compare_policy_timing.py")
    comparison_spec = importlib.util.spec_from_file_location("compare_policy_timing", script)
    comparison = importlib.util.module_from_spec(comparison_spec)
    comparison_spec.loader.exec_module(comparison)
    torch.set_num_threads(2)
    cfg = tiny()
    model = PersonalModel(cfg).eval()
    policy = tmp_path / "policy.pt"
    torch.save(model.state_dict(), policy)
    head = tmp_path / "timing.pt"
    groups = ["clock"]
    timing = TimingHead(TimingHeadConfig(side_dim=len(feature_names(groups)), pooled_dim=cfg.dim_vit,
                                        move_mode="squares"))
    save_timing_head(head, timing, base_checkpoint_sha256=rebind_module.digest(policy),
                     feature_names=feature_names(groups), extra={"feature_groups": groups})
    positions = tmp_path / "val.jsonl"
    example = dict(row(), schema_version="timing-data-v2", game_id="game", ply=1,
                   observation_interval_seconds=[1.4, 1.6])
    positions.write_text(json.dumps(example) + "\n")

    def load(path, device):
        restored = PersonalModel(cfg).eval()
        restored.load_state_dict(torch.load(path, weights_only=True))
        return restored.to(device)

    monkeypatch.setattr(comparison, "load_model", load)
    report = comparison.compare(positions, policy, policy, head, head, tmp_path / "report.json", crps_points=32)
    assert report["candidate_minus_original"]["crps_seconds"]["difference"] == 0
    assert report["candidate_minus_original"]["mean_absolute_error"]["difference"] == 0
    assert report["scores"]["candidate"]["groups"]["all"]["timing_labels"] == 1
    assert report["scores"]["candidate"]["timing_information"] == "human_move_conditional_diagnostic"
