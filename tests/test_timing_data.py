import copy
import json
from pathlib import Path

import pytest

from chess_self.data import prepare_pgn
from chess_self.timing_data import (GROUPS, SCHEMA_VERSION, chronological_partition,
    feature_names, feature_vector, prepare_timing_data, validate_public_rows)

MOVES = '1. e4 {[%clk 0:01:39]} e5 {[%clk 0:01:38]} 2. Nf3 {[%clk 0:01:35.0]} Nc6 {[%clk 0:01:36]} 3. Bb5 {[%clk 0:01:30.0]} a6 {[%clk 0:01:32]} *'


def pgn(i, moves=MOVES, **extra):
    headers = dict(Event='test', UTCDate=f'2025.01.{i:02}', UTCTime='12:00:00',
        White='Me', Black=f'Opponent{i}', Result='*', WhiteElo='1500', BlackElo='1600',
        TimeControl='100+2', Site='https://lichess.org/example', Termination='Normal')
    headers.update(extra)
    return '\n'.join(f'[{k} "{v}"]' for k, v in headers.items()) + '\n\n' + moves + '\n\n'


def prepared(tmp_path, games=None):
    path = tmp_path/'games.pgn'
    path.write_text(''.join(games or [pgn(i) for i in range(1, 5)]))
    reference = tmp_path/'reference'
    prepare_pgn([path], ['me'], reference)
    # Prove the new pipeline never opens original test examples.
    (reference/'test.jsonl').write_text('not valid JSON: do not read this')
    output = tmp_path/'v2'
    report = prepare_timing_data([path], ['me'], reference, output)
    rows = {s:[json.loads(v) for v in (output/f'{s}.jsonl').read_text().splitlines()] for s in ('train','val')}
    return path, reference, output, report, rows


def test_preserves_benchmark_and_excludes_test(tmp_path):
    _, ref, out, manifest, rows = prepared(tmp_path)
    assert manifest['schema_version'] == SCHEMA_VERSION
    assert not (out/'test.jsonl').exists()
    assert json.loads((out/'manifest.json').read_text()) == manifest
    for split in ('train','val'):
        old = [json.loads(v) for v in (ref/f'{split}.jsonl').read_text().splitlines()]
        assert len(old) == len(rows[split])
        for legacy, new in zip(old, rows[split]):
            assert {k:new[k] for k in legacy} == legacy
    reserved = set(manifest['reserved_test_game_ids'])
    assert all(r['game_id'] not in reserved for rs in rows.values() for r in rs)


def test_clocks_histories_precision_and_metadata(tmp_path):
    _, _, _, _, rows = prepared(tmp_path)
    one, three, five = rows['train'][:3]
    assert one['opponent_remaining_seconds'] == 100
    assert one['own_time_history'] == []
    assert three['opponent_remaining_seconds'] == 98
    assert three['own_time_history'] == [None]
    assert three['opponent_time_history'] == [None]
    assert five['own_time_history'] == [None,6.0]
    assert five['opponent_time_history'] == [None,4.0]
    assert three['clock_precision_before_seconds'] == 1
    assert three['clock_precision_after_seconds'] == .1
    assert three['observation_interval_seconds'] == [5.45,6.55]
    assert three['observation_interval_truncation_sensitivity_seconds'] == [5.9,7.]
    assert one['observation_interval_seconds'] is None
    assert three['timestamp_utc'] == '2025-01-01T12:00:00+00:00'
    assert three['timestamp_uncertain'] is False
    assert three['platform'] == 'lichess'
    assert three['explicit_premove'] is None
    assert three['canonical_time_control'] == '100+2'


def test_missing_opponent_clock_not_forward_filled(tmp_path):
    moves = MOVES.replace('Nc6 {[%clk 0:01:36]}', 'Nc6')
    _, _, _, _, rows = prepared(tmp_path, [pgn(i,moves) for i in range(1,5)])
    five = rows['train'][2]
    assert five['opponent_remaining_seconds'] is None
    assert five['opponent_clock_known'] is False
    assert five['opponent_time_history'] == [None,None]


def test_invalid_opponent_clock_clears_bank(tmp_path):
    moves = MOVES.replace('0:01:36', '0:01:43')
    _, _, _, _, rows = prepared(tmp_path, [pgn(i,moves) for i in range(1,5)])
    assert rows['train'][2]['opponent_remaining_seconds'] is None
    assert rows['train'][2]['opponent_time_history'] == [None,None]


def test_current_label_and_move_do_not_change_features(tmp_path):
    _, _, _, _, rows = prepared(tmp_path)
    row = rows['train'][1]
    changed = dict(row, elapsed_seconds=999, observation_interval_seconds=[998,1000],
        clock_precision_after_seconds=100, move_uci='e1e8', termination='Time forfeit', result='0-1')
    groups = list(GROUPS)
    assert feature_vector(row,groups) == feature_vector(changed,groups)
    assert len(feature_vector(row,groups)) == len(feature_names(groups))
    with pytest.raises(ValueError):feature_names(['clock','clock'])
    with pytest.raises(ValueError):feature_names(['post_clock'])


def test_changed_current_clock_label_does_not_enter_current_features(tmp_path):
    a = tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    _,_,_,_,old=prepared(a)
    changed = MOVES.replace('0:01:35.0','0:01:31.0')
    _,_,_,_,new=prepared(b,[pgn(i,changed) for i in range(1,5)])
    assert old['train'][1]['elapsed_seconds'] != new['train'][1]['elapsed_seconds']
    assert feature_vector(old['train'][1],list(GROUPS)) == feature_vector(new['train'][1],list(GROUPS))


def test_familiarity_only_earlier_training_games(tmp_path):
    _,_,_,_,rows=prepared(tmp_path)
    assert rows['train'][0]['prior_position_games'] == 0
    assert rows['train'][3]['prior_position_games'] == 1
    assert rows['val'][0]['prior_position_games'] == 2
    assert rows['train'][0]['first_unfamiliar_position'] is True
    assert rows['train'][1]['first_unfamiliar_position'] is False


def test_unknown_same_day_order_does_not_leak(tmp_path):
    games=[pgn(i, UTCDate='2025.01.01', UTCTime='?') for i in range(1,5)]
    _,_,_,_,rows=prepared(tmp_path,games)
    assert all(r['timestamp_uncertain'] for rs in rows.values() for r in rs)
    assert all(r['prior_position_games']==0 for rs in rows.values() for r in rs)


def test_does_not_overwrite_and_detects_changed_source(tmp_path):
    source,ref,out,_,_=prepared(tmp_path)
    with pytest.raises(ValueError,match='empty output'):prepare_timing_data([source],['me'],ref,out)
    source.write_text(source.read_text().replace('0:01:35.0','0:01:31.0'))
    with pytest.raises(ValueError,match='annotations differ'):
        prepare_timing_data([source],['me'],ref,tmp_path/'changed')
    assert not (tmp_path/'changed').exists()


def test_chronological_partition_groups_sessions_and_blocks_reserved():
    def row(g,stamp):return {'game_id':g,'ply':1,'timestamp_utc':stamp,'timestamp_uncertain':False}
    rows=[row('a','2025-01-01T10:00:00Z'),row('b','2025-01-02T11:50:00Z'),row('c','2025-01-02T12:10:00Z')]
    split=chronological_partition(rows,'2025-01-02T12:00:00Z')
    assert [r['game_id'] for r in split['train']]==['a']
    assert [r['game_id'] for r in split['val']]==['b','c']
    with pytest.raises(ValueError,match='Reserved'):chronological_partition(rows,'2025-01-02T12:00:00Z',reserved_game_ids=['c'])
    rows[0]['timestamp_uncertain']=True
    with pytest.raises(ValueError,match='timestamps'):chronological_partition(rows,'2025-01-02T12:00:00Z')


def test_public_prior_identity_cutoff_and_duplicate_guards():
    row={'game_id':'a','ply':1,'white_player':'alice','black_player':'bob','timestamp_utc':'2025-01-01T12:00:00Z','timestamp_uncertain':False}
    opts={'target_accounts':['Me'],'reserved_game_ids':['reserved'],'cutoff':'2025-02-01T00:00:00Z'}
    assert validate_public_rows([row],**opts)==[row]
    for change,match in [({'white_player':'ME'},'Target account'),({'game_id':'reserved'},'Reserved'),({'timestamp_utc':None},'timestamps'),({'timestamp_utc':'2025-02-01T00:00:00Z'},'cutoff'),({'white_player':None},'identities')]:
        with pytest.raises(ValueError,match=match):validate_public_rows([dict(row,**change)],**opts)
    with pytest.raises(ValueError,match='Duplicate'):validate_public_rows([row,row],**opts)


def test_retrieval_uses_only_earlier_other_training_games():
    import math
    import numpy as np
    from chess_self.timing_data import retrieval_features

    def row(g, date, elapsed, stamp=None):
        return dict(game_id=g, date=date, elapsed_seconds=elapsed, timestamp_utc=stamp)
    training = [row('earlier','2025-01-01',3), row('same','2025-01-02',999),
                row('future','2025-01-03',999), row('ambiguous','2025-01-02',999),
                row('missing','2025-01-01',None), row('earlier2','2025-01-01',15)]
    query = [row('same','2025-01-02',123)]
    train_emb = np.array([[1.,0.],[1.,0.],[1.,0.],[1.,0.],[1.,0.],[0.,1.]])
    query_emb = np.array([[1.,0.]])
    nearest = retrieval_features(training,train_emb,query,query_emb,k=1)
    assert len(nearest) == 1
    assert nearest[0]['neighbor_count'] == 1
    assert nearest[0]['nearest_timing_logmean'] == pytest.approx(math.log(4), rel=1e-14, abs=1e-14)
    result = retrieval_features(training,train_emb,query,query_emb,k=8)
    assert result[0]['neighbor_count'] == 2
    assert result[0]['nearest_timing_logmean'] == pytest.approx((math.log(4)+math.log(16))/2)
    query[0]['elapsed_seconds'] = 999999
    assert retrieval_features(training,train_emb,query,query_emb,k=8) == result
    assert feature_vector(result[0],['retrieval']) == pytest.approx(
        [result[0]['nearest_timing_logmean'],math.log(3)], rel=1e-14, abs=1e-14)


def test_retrieval_timestamp_order_empty_candidates_and_shape_guards():
    from chess_self.timing_data import retrieval_features
    training = [{'game_id':'a','date':'2025-01-01','timestamp_utc':'2025-01-01T12:00:00+00:00','elapsed_seconds':5}]
    query = [{'game_id':'b','date':'2025-01-01','timestamp_utc':'2025-01-01T11:00:00+00:00'}]
    assert retrieval_features(training,[[1,0]],query,[[1,0]])[0]['neighbor_count']==0
    query[0]['timestamp_utc']='2025-01-01T13:00:00+00:00'
    assert retrieval_features(training,[[1,0]],query,[[1,0]])[0]['neighbor_count']==1
    for bad in (0,-1,True,1.2):
        with pytest.raises(ValueError):retrieval_features(training,[[1,0]],query,[[1,0]],k=bad)
    with pytest.raises(ValueError,match='dimension'):retrieval_features(training,[[1,0]],query,[[1]])
    with pytest.raises(ValueError,match='match'):retrieval_features(training,[[1,0],[0,1]],query,[[1,0]])
    with pytest.raises(ValueError,match='Nonfinite'):retrieval_features(training,[[float('nan'),0]],query,[[1,0]])
