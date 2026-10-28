import importlib.util
from pathlib import Path
from types import SimpleNamespace

import chess
import chess.engine
import pytest
import torch

spec = importlib.util.spec_from_file_location('play_policy', Path(__file__).parents[1] / 'scripts/play_policy_candidates.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Engine:
    def __init__(self):
        self.clears = 0

    def configure(self, options):
        assert options == {'Clear Hash': None}
        self.clears += 1

    def analyse(self, board, limit, root_moves=None):
        move = root_moves[0] if root_moves else next(iter(board.legal_moves))
        return {'score': chess.engine.PovScore(chess.engine.Cp(-300 if root_moves else 0), board.turn), 'pv': [move]}

    def play(self, board, limit):
        return SimpleNamespace(move=next(iter(board.legal_moves)))


def test_seeded_sampling_is_repeatable():
    policy = [{'uci': 'e2e4', 'probability': .6}, {'uci': 'd2d4', 'probability': .4}]
    def run():
        generator = torch.Generator().manual_seed(100)
        return [module.sample_move(policy, generator) for _ in range(20)]
    assert run() == run()
    assert {move.uci() for move, _, _ in run()} == {'e2e4', 'd2d4'}


@pytest.mark.parametrize('color', [chess.WHITE, chess.BLACK])
def test_bounded_games_use_pre_move_context_and_fixed_clocks(monkeypatch, color):
    rows = []
    def predict(model, row):
        rows.append(row)
        board = chess.Board(row['fen'])
        move = next(iter(board.legal_moves))
        return {'moves': [{'uci': move.uci(), 'probability': 1.}]}
    monkeypatch.setattr(module, 'predict', predict)
    engine = Engine()
    game, records = module.play_game(None, engine, seed=100, color=color, max_plies=6)
    assert game.headers['PlyCount'] == '6'
    assert game.headers['Termination'] == 'ply cap'
    assert len(records) == 3
    assert [r['remaining_seconds'] for r in rows] == [600., 598., 596.]
    assert all(r['history_fens'][-1] == r['fen'] for r in rows)
    assert all(r['elo_self'] == 680 and r['elo_oppo'] == 680 for r in rows)
    assert all(r['loss_cp'] == 300 for r in records)
    assert engine.clears == 9
    assert module.summarize(records)['loss_ge_200cp'] == 3
    assert module.summarize(records)['loss_ge_500cp'] == 0


def test_cli_rejects_existing_output_and_unbounded_games(tmp_path):
    with pytest.raises(SystemExit):
        module.main(['--checkpoint', 'x.pt', '--output', str(tmp_path)])
    with pytest.raises(SystemExit):
        module.main(['--checkpoint', 'x.pt', '--output', str(tmp_path / 'new'), '--games', '1000'])


def material_record(opening, move, reply, loss=300):
    board = chess.Board()
    for san in opening.split():
        board.push_san(san)
    selected = board.parse_san(move)
    fen = board.fen()
    board.push(selected)
    response = board.parse_san(reply)
    return {'fen': fen, 'uci': selected.uci(), 'pv': [selected.uci(), response.uci()], 'loss_cp': loss}


@pytest.mark.parametrize('opening,move,reply,expected', [
    ('e4 e5 Nc3 Nf6', 'Qh5', 'Nxh5', True),
    ('e4 e5', 'Nf3', 'Nc6', False),
    ('e4 d6 d4 Nc6 d5', 'Nd4', 'Qxd4', True),
])
def test_direct_material_screen(opening, move, reply, expected):
    record = material_record(opening, move, reply)
    result = module.annotate_material_loss(record)
    assert result['direct_material_loss'] is expected
    assert 'direct_material_loss' not in record
    assert module.annotate_material_loss(dict(record, loss_cp=199))['direct_material_loss'] is False


def test_recapture_removes_apparent_gain_and_board_is_unchanged():
    record = {'fen': '4k3/8/8/2b5/3N4/8/8/3RK3 w - - 0 1',
              'uci': 'e1f1', 'pv': ['e1f1', 'c5d4'], 'loss_cp': 300}
    # Rxd4 recovers the bishop after Bxd4 takes the knight.
    result = module.annotate_material_loss(record)
    assert result['direct_material_loss'] is False
    board = chess.Board('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1')
    assert module.capture_gain(board, chess.Move.from_uci('e5d6')) == 1
    promotion = chess.Board('r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1')
    assert module.capture_gain(promotion, chess.Move.from_uci('b7a8q')) == 13
    before = promotion.fen()
    module.exchange_gain(promotion, chess.A8)
    assert promotion.fen() == before
