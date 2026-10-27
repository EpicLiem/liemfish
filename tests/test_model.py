import chess
import pytest
import torch
from maia3.dataset import get_historical_tokens, tokenize_board
from maia3.models import MAIA3Model
from chess_self.model import PersonalModel, config, encode, MOVE_INDEX, predict
from chess_self.training import loss_for, select_parameters


def row(moves=()):
    board = chess.Board()
    history = [board.fen()]
    for move in moves:
        board.push_uci(move)
        history.append(board.fen())
    return dict(fen=board.fen(), history_fens=history[-8:], elo_self=700, elo_oppo=900,
                initial_seconds=180, increment_seconds=2, remaining_seconds=7,
                elapsed_seconds=1.5, move_uci=next(iter(board.legal_moves)).uci())


def tiny():
    cfg = config()
    cfg.dim_vit = 32
    cfg.dim_emb = 8
    cfg.head_hid_dim = 16
    cfg.num_blocks = 1
    cfg.num_heads = 2
    cfg.gab_gen_size = 4
    cfg.gab_per_square_dim = 2
    cfg.gab_intermediate_dim = 8
    return cfg


def test_history_matches_pinned_upstream_and_excludes_elapsed():
    r = row(["e2e4", "e7e5", "g1f3"])
    encoded = encode(r)
    expected = get_historical_tokens([tokenize_board(chess.Board(fen)) for fen in r["history_fens"]],
                                    config(), base=180, inc=2, clk_left_before=7, clk_ponder=0)[:, :99]
    torch.testing.assert_close(encoded["tokens"], expected)
    r["elapsed_seconds"] = 85
    torch.testing.assert_close(encode(r)["tokens"], encoded["tokens"])
    assert encoded["tokens"].shape == (64, 99)


def test_black_promotion_and_legal_mask():
    fen = "7k/8/8/8/8/8/p7/7K b - - 0 1"
    r = row()
    r.update(fen=fen, history_fens=[fen], move_uci="a2a1n")
    encoded = encode(r)
    assert encoded["target"] == MOVE_INDEX["a7a8n"]
    assert encoded["legal"][encoded["target"]]


@pytest.mark.parametrize("elapsed,valid", [(None, False), (-1, False), (0, True), (99.8, True), (99.9, False), (200, False)])
def test_timing_labels_outside_support_are_not_clamped(elapsed, valid):
    r = row()
    r["elapsed_seconds"] = elapsed
    assert bool(encode(r)["time_valid"]) == valid


def test_wrapper_parity_and_optimizer_changes_only_selected_weights():
    torch.set_num_threads(2)
    cfg = tiny()
    model = PersonalModel(cfg).eval()
    original = MAIA3Model(cfg).eval()
    original.load_state_dict(model.state_dict())
    batch = {k: v.unsqueeze(0) for k, v in encode(row()).items()}
    args = (batch["tokens"], batch["self_elos"], batch["oppo_elos"])
    out = model(*args)
    upstream = original(*args)
    torch.testing.assert_close(out[0], upstream[0])
    torch.testing.assert_close(out[2], upstream[2])
    torch.testing.assert_close((out[1].softmax(-1) * model.ponder_bucket_centers).sum(-1), out[2])
    parameters = select_parameters(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    loss = loss_for(model, batch, .2)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before["proj_sq_from.weight"], model.proj_sq_from.weight)
    assert not torch.equal(before["fc_ponder.weight"], model.fc_ponder.weight)
    torch.testing.assert_close(before["token_projection.weight"], model.token_projection.weight)
    result = predict(model, row())
    assert abs(sum(item["probability"] for item in result["moves"]) - 1) < 1e-5
    assert all(chess.Move.from_uci(item["uci"]) in chess.Board().legal_moves for item in result["moves"])
