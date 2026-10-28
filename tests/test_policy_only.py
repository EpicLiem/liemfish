import json

import pytest
import torch

from chess_self.cli import main
from chess_self.model import PersonalModel, encode
from chess_self.timing_model import MaiaFeatures
from chess_self.training import loss_for, select_parameters, train
from test_model import row, tiny


def test_policy_step_preserves_timing_features_and_every_other_weight():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = PersonalModel(tiny()).eval()
    extractor = MaiaFeatures(model)
    parameters = select_parameters(model, policy_only=True)
    selected = {name for name, value in model.named_parameters() if value.requires_grad}
    assert selected == {"proj_sq_from.weight", "proj_sq_to.weight", "promo_bias_proj.weight"}
    batch = {key: value.unsqueeze(0) for key, value in encode(row()).items()}
    inputs = (batch["tokens"], batch["self_elos"], batch["oppo_elos"])
    before_weights = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        before_features = {key: value.clone() for key, value in extractor(*inputs).items()}
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    loss_for(model, batch, 0).backward()
    optimizer.step()
    with torch.no_grad():
        after_features = extractor(*inputs)
    for key in ("squares", "pooled", "base_logits"):
        torch.testing.assert_close(before_features[key], after_features[key], rtol=0, atol=0)
    assert not torch.equal(before_features["move_logits"], after_features["move_logits"])
    assert not torch.equal(before_weights["proj_sq_from.weight"], model.proj_sq_from.weight)
    for name, value in model.state_dict().items():
        if name not in selected:
            torch.testing.assert_close(before_weights[name], value, rtol=0, atol=0)
    extractor.close()


@pytest.mark.parametrize("options", [{"timing_weight": .2}, {"timing_weight": 0, "final_blocks": 1}])
def test_policy_only_rejects_incompatible_training_before_reading_data(tmp_path, options):
    with pytest.raises(ValueError, match="Policy-only training requires"):
        train(tmp_path / "absent-data", tmp_path / "absent-checkpoint", tmp_path / "output",
              device="cpu", policy_only=True, **options)
    assert not (tmp_path / "output").exists()


def test_policy_only_selection_rejects_backbone_adaptation():
    with pytest.raises(ValueError, match="cannot adapt"):
        select_parameters(PersonalModel(tiny()), final_blocks=1, policy_only=True)


def test_cli_passes_policy_only(monkeypatch, tmp_path):
    called = {}

    def fake_train(**options):
        called.update(options)
        return {"selected_epoch": 0, "optimizer_steps": 0}

    monkeypatch.setattr("chess_self.training.train", fake_train)
    main(["train", "--device", "cpu", "--output", str(tmp_path / "run"),
          "--policy-only", "--timing-weight", "0"])
    assert called["policy_only"] is True
    assert called["timing_weight"] == 0
    assert called["final_blocks"] == 0


def test_smoke_records_policy_only_and_keeps_backbone_in_eval(monkeypatch, tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train", "val"):
        example = dict(row(), game_id=name, time_control="180+2", player_color="white", ply=1)
        (data / f"{name}.jsonl").write_text(json.dumps(example) + "\n")
    (data / "test.jsonl").write_text("")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    model = PersonalModel(tiny()).eval()
    monkeypatch.setattr("chess_self.training.load_model", lambda *args: model)
    modes = []
    handle = model.transformer.register_forward_pre_hook(lambda module, inputs: modes.append(module.training))
    report = train(data, checkpoint, tmp_path / "run", device="cpu", smoke=True,
                   policy_only=True, timing_weight=0)
    handle.remove()
    assert modes and not any(modes)
    assert report["settings"]["policy_only"] is True
    saved = json.loads((tmp_path / "run/run.json").read_text())
    assert saved["settings"]["policy_only"] is True
    assert saved["optimizer_steps"] == 1
