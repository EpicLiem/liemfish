import json
from pathlib import Path

import chess
import pytest
import torch

from chess_self.model import encode
from chess_self.timing_cache import FeatureCache, digest, write_json, assert_comparable
from chess_self.timing_experiments import (timing_objective, sample_weights, train_experiment,
                                          predict_cache, move_metrics)
from chess_self.timing_calibration import assert_unseen_predictor_games, fit_calibration
from chess_self.timing_events import event_data, fit_events
from chess_self.timing_data import feature_names
from chess_self.timing_model import TimingHead, TimingHeadConfig


def cache_fixture(path, prefix='train', n=4, fitted=None):
    path.mkdir()
    board=chess.Board(); rows=[]
    for i in range(n):
        fen=board.fen(); move=next(iter(board.legal_moves))
        rows.append({'schema_version':'timing-data-v2','game_id':f'{prefix}-{i//2}',
                     'date':f'2024-01-{i//2+1:02d}','ply':i+1,'player_color':'white' if board.turn else 'black',
                     'fen':fen,'history_fens':[fen],'move_uci':move.uci(),'elo_self':900,'elo_oppo':1000,
                     'initial_seconds':60.,'increment_seconds':0.,'remaining_seconds':40.,
                     'elapsed_seconds':2.+i,'observation_interval_seconds':[1.9+i,2.1+i],
                     'own_time_history':[],'opponent_time_history':[],'time_control':'60'})
        board.push(move)
    logits=torch.randn(n,4352)
    for i,r in enumerate(rows):logits[i].masked_fill_(~encode(r)['legal'],-torch.inf)
    torch.save({'pooled':torch.randn(n,8),'squares':torch.randn(n,64,8).half(),
                'base_logits':torch.zeros(n,30),'move_logits':logits},path/'features.pt')
    (path/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_json(path/'manifest.json',{'cache_version':1,'checkpoint_sha256':'test-sha',
        'count':n,'smoke_only':True,'predictor_training_game_ids':fitted,
        'rows_sha256':digest(path/'rows.jsonl'),
        'shards':[{'file':'features.pt','start':0,'count':n,'sha256':digest(path/'features.pt')}]})
    return FeatureCache(path)


def test_cache_integrity_and_overlap(tmp_path):
    a=cache_fixture(tmp_path/'a');b=cache_fixture(tmp_path/'b')
    with pytest.raises(ValueError,match='overlap'):assert_comparable(a,b)
    (tmp_path/'a'/'features.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):FeatureCache(tmp_path/'a')


def test_training_interval_loss_does_not_discard_low_probability():
    raw=torch.zeros(2,30);raw[:,0]=30;raw.requires_grad_()
    rows=[{'elapsed_seconds':10.,'remaining_seconds':60.,'observation_interval_seconds':[9.9,10.1]}]*2
    loss,used,excluded=timing_objective(raw,rows,'absolute')
    assert used==2 and excluded==0 and loss>25
    loss.backward();assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0


@pytest.mark.parametrize('family,width',[('absolute',30),('extended',40),('fraction',30),('beta',2)])
def test_family_training_step(family,width):
    raw=torch.zeros(3,width,requires_grad=True)
    rows=[{'elapsed_seconds':v,'remaining_seconds':60.,'observation_interval_seconds':[max(0,v-.1),v+.1]} for v in [0.,2.,40.]]
    loss,n,excluded=timing_objective(raw,rows,family)
    assert n==3 and excluded==0 and torch.isfinite(loss)
    loss.backward();assert torch.isfinite(raw.grad).all()


def test_unsupported_label_is_counted():
    raw=torch.zeros(2,30,requires_grad=True)
    rows=[{'elapsed_seconds':v,'remaining_seconds':300.,'observation_interval_seconds':[v-.1,v+.1]} for v in [2.,180.]]
    loss,n,excluded=timing_objective(raw,rows,'absolute')
    assert n==1 and excluded==1


def test_replacement_initialization_never_deployed_without_reference_improvement(tmp_path):
    torch.set_num_threads(1)
    cache_fixture(tmp_path/'train');cache_fixture(tmp_path/'val','val')
    config={'features':['clock'],'head':{'pooled_dim':8,'hidden_dim':4,'output_family':'fraction'},
            'max_move_nll_regression':-1.}
    report=train_experiment(tmp_path/'train',tmp_path/'val',tmp_path/'run',config,epochs=1,smoke=True)
    assert report['selected_epoch']==0 and not report['accepted_candidate']
    assert (tmp_path/'run'/'initial.pt').exists() and not (tmp_path/'run'/'best.pt').exists()


def test_marginal_does_not_use_future_human_move(tmp_path):
    torch.set_num_threads(1)
    cache=cache_fixture(tmp_path/'val','val',n=2)
    head=TimingHead(TimingHeadConfig(side_dim=len(feature_names(['clock'])),pooled_dim=8,hidden_dim=4,move_mode='attributes'))
    with torch.no_grad():head.output.weight.normal_()
    dist,logits,_=predict_cache(head,cache,['clock'],batch_size=2)
    first=dist.mean()
    for r in cache.rows:r['move_uci']=list(chess.Board(r['fen']).legal_moves)[-1].uci()
    second,_,_=predict_cache(head,cache,['clock'],batch_size=2)
    torch.testing.assert_close(first,second.mean())
    assert torch.isfinite(first).all()


def test_calibration_refuses_training_and_unknown_provenance(tmp_path):
    a=cache_fixture(tmp_path/'a')
    with pytest.raises(ValueError,match='provenance'):assert_unseen_predictor_games(a)
    b=cache_fixture(tmp_path/'b',fitted=['train-0'])
    with pytest.raises(ValueError,match='in-sample'):assert_unseen_predictor_games(b)
    c=cache_fixture(tmp_path/'c','heldout',fitted=['train-0'])
    assert_unseen_predictor_games(c)


def test_calibration_end_to_end(tmp_path):
    torch.set_num_threads(1)
    cache_fixture(tmp_path/'fit','calibration',fitted=['earlier'])
    cache_fixture(tmp_path/'val','val',fitted=['earlier'])
    report=fit_calibration(tmp_path/'fit',tmp_path/'val',tmp_path/'run',epochs=1)
    assert report['calibrated']['groups']['all']['positions']==4
    assert (tmp_path/'run'/'calibration.pt').exists()


def test_recency_and_balancing():
    rows=[{'date':'2023-01-01','initial_seconds':60,'increment_seconds':0},
          {'date':'2024-01-01','initial_seconds':300,'increment_seconds':0},
          {'date':'2024-01-01','initial_seconds':300,'increment_seconds':0}]
    assert sample_weights(rows,'recency')[0]<sample_weights(rows,'recency')[1]
    assert sample_weights(rows,'balanced').tolist()==[1.,.5,.5]


def test_timeout_labels_require_verified_censoring(tmp_path):
    c=cache_fixture(tmp_path/'c')
    rows=c.rows
    for i,r in enumerate(rows):
        r['event_type']='timeout' if i%2 else 'completed';r['event_verified']=True
        if i%2:r['elapsed_seconds']=None
    chosen,x,y,_=event_data(rows)
    assert y.tolist()==[0,1,0,1]
    rows[1]['elapsed_seconds']=40
    with pytest.raises(ValueError,match='censored'):event_data(rows)


def test_required_augmentation_cannot_silently_become_zero(tmp_path):
    from chess_self.timing_experiments import side_inputs
    cache=cache_fixture(tmp_path/'cache')
    for group in ('engine','retrieval','familiarity'):
        with pytest.raises(ValueError,match='Missing causal'):
            side_inputs(cache.rows,[group],'cpu')


def test_public_training_requires_checked_source(tmp_path):
    cache_fixture(tmp_path/'train');cache_fixture(tmp_path/'val','val')
    with pytest.raises(ValueError,match='public-check'):
        train_experiment(tmp_path/'train',tmp_path/'val',tmp_path/'run',
                         {'public_training':True},smoke=True)


def test_replay_preserves_supplied_retrieval_and_engine_features(tmp_path):
    from chess_self.timing_runtime import recorded_replay
    cache=cache_fixture(tmp_path/'cache',n=1)
    row=cache.rows[0]
    row.update(nearest_timing_logmean=1.5,neighbor_count=8,engine_score_disagreement=.5,engine_move_disagreement=1.)
    def predictor(board,context):
        assert context['nearest_timing_logmean']==1.5
        assert context['engine_score_disagreement']==.5
        assert 'elapsed_seconds' not in context and 'move_uci' not in context
        return {'move_uci':next(iter(board.legal_moves)).uci(),'target_seconds':1.}
    recorded_replay([row],predictor)


def test_event_fit_smoke(tmp_path):
    a=cache_fixture(tmp_path/'a','a');b=cache_fixture(tmp_path/'b','b')
    for cache in (a,b):
        for i,row in enumerate(cache.rows):
            row['event_type']='timeout' if i%2 else 'completed';row['event_verified']=True
            if i%2:row['elapsed_seconds']=None
        (cache.path/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cache.rows))
    report=fit_events(a.path/'events.jsonl',b.path/'events.jsonl',tmp_path/'run',epochs=2)
    assert 0<=report['validation_brier']<=1
    assert report['duration_model'].startswith('separate')
