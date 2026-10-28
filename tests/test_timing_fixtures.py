import json
from types import SimpleNamespace

import chess
import pytest
import torch

from chess_self import timing_fixtures
from chess_self.export import parity_rows
from chess_self.model import PersonalModel, config
from chess_self.timing_cache import digest
from chess_self.timing_model import TimingHead, TimingHeadConfig, save_timing_head


def fixture_rows():
    rows = parity_rows()
    for i, row in enumerate(rows):
        board = chess.Board(row['fen'])
        row.update(game_id=f'game-{i}', ply=board.ply(),
                   player_color='white' if board.turn else 'black',
                   move_uci=next(iter(board.legal_moves)).uci())
    return [rows[1], rows[0], *rows[2:]]


def test_selection_covers_explicit_colors_when_first_row_is_black():
    rows = fixture_rows()
    white, black, low = timing_fixtures.select_fixture_rows(rows)
    assert white['player_color'] == 'white'
    assert black['player_color'] == 'black'
    assert len({row['fen'] for row in (white, black, low)}) == 3
    assert low['remaining_seconds'] == min(row['remaining_seconds'] for row in rows)


@pytest.mark.parametrize('missing', ['white', 'black'])
def test_selection_refuses_missing_color(missing):
    rows = [row for row in fixture_rows() if row['player_color'] != missing]
    with pytest.raises(ValueError, match=f'explicit {missing}'):
        timing_fixtures.select_fixture_rows(rows)


def test_selection_checks_orientation_and_clocks():
    rows = fixture_rows()
    rows[0]['player_color'] = 'white'
    with pytest.raises(ValueError, match='FEN side'):
        timing_fixtures.select_fixture_rows(rows)
    rows = fixture_rows()
    rows[0]['remaining_seconds'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        timing_fixtures.select_fixture_rows(rows)


def test_export_records_real_case_provenance_and_matching_scheduler(tmp_path, monkeypatch):
    cfg = vars(config()).copy()
    cfg.update(dim_vit=32, dim_emb=8, head_hid_dim=16, num_blocks=1,
               num_heads=4, gab_gen_size=8, gab_per_square_dim=4, gab_intermediate_dim=8)
    torch.manual_seed(83)
    native = PersonalModel(SimpleNamespace(**cfg)).eval()
    checkpoint = tmp_path/'tiny.pt'
    torch.save(native.state_dict(), checkpoint)
    monkeypatch.setattr(timing_fixtures, 'load_model', lambda path: native)
    head = TimingHead(TimingHeadConfig(side_dim=0, pooled_dim=32, move_mode='squares'))
    head_path = tmp_path/'head.pt'
    save_timing_head(head_path, head, base_checkpoint_sha256=digest(checkpoint),
                     feature_names=[], extra={'feature_groups': [], 'smoke_only': False})
    rows = fixture_rows()
    positions = tmp_path/'val.jsonl'
    positions.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    output = tmp_path/'fixtures.json'
    result = timing_fixtures.export_browser_fixtures(checkpoint, head_path, positions, output)
    report = json.loads(output.read_text())
    assert result['cases'] == 4
    cases = {case['name']: case for case in report['cases']}
    assert cases['white']['positions'][0]['player_color'] == 'white'
    assert cases['black']['positions'][0]['player_color'] == 'black'
    assert cases['white']['positions'][0]['game_id'] != cases['black']['positions'][0]['game_id']
    assert cases['batch-two']['positions'] == cases['white']['positions'] + cases['black']['positions']
    assert cases['batch-two']['expected']['timing_raw']['shape'] == [2, 30]
    assert report['scheduler']['position_key'] == cases[report['scheduler']['case_name']]['positions'][0]['fen']
    assert report['config']['head_checkpoint_sha256'] == digest(head_path)
    assert report['config']['positions_sha256'] == digest(positions)
    assert report['config']['smoke_only'] is False
    assert report['browser_execution_verified'] is False
