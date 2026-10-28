import json
import pytest
import torch
from chess_self.timing_distributions import BinnedDistribution
from chess_self.timing_metrics import scorecard, paired_game_bootstrap


def row(game,time,clock=10.,control=60):
    return dict(game_id=game,ply=4,elapsed_seconds=time,remaining_seconds=clock,initial_seconds=control,increment_seconds=0.,platform='test',observation_interval_seconds=None if time is None else [max(0,time-.1),time+.1])


def test_scorecard_common_observation_intervals_and_zero_support():
    d=BinnedDistribution(torch.zeros(3,1),[0.,10.],'linear')
    rows=[row('a',5),row('b',180),row('b',None)]
    r=scorecard(d,rows)
    all_=r['groups']['all']
    assert all_['timing_labels']==2 and all_['positions']==3
    assert all_['zero_probability_intervals']==1
    assert all_['interval_nll'] is None
    assert all_['finite_interval_nll']==pytest.approx(-torch.tensor(.02).log().item(),abs=1e-5)
    assert all_['outside_model_support']==1
    assert all_['common_support_labels']==1
    json.dumps(r,allow_nan=False)


def test_missing_precision_is_not_invented_and_groups_are_canonical():
    rows=[row('a',1),row('b',2)]
    rows[0]['time_control']='60'; rows[1]['time_control']='60+0'
    rows[0]['observation_interval_seconds']=None
    r=scorecard(BinnedDistribution(torch.zeros(2,1),[0.,10.],'linear'),rows)
    assert r['groups']['all']['missing_observation_intervals']==1
    assert list(k for k in r['groups'] if k.startswith('control/'))==['control/60+0']
    assert r['groups']['clock/10_to_30s']['positions']==2


def test_truncation_and_macro_weights():
    rows=[row('a',1,2,60),row('b',2,10,600)]
    d=BinnedDistribution(torch.zeros(2,1),[0.,10.],'linear').truncate(torch.tensor([2.,10.]))
    r=scorecard(d,rows,deployment_weights={'60+0':1.,'600+0':0.})
    assert r['groups']['all']['impossible_duration_mass']==pytest.approx(0)
    assert r['groups']['all']['removed_mass']==pytest.approx(.4)
    assert r['deployment_weighted']['mean_mae_seconds']==pytest.approx(0,abs=1e-5)
    with pytest.raises(ValueError):scorecard(d,rows,deployment_weights={'30+0':1})


def test_bootstrap_whole_games_and_nonfinite_pairs():
    rows=[row('a',1),row('a',2),row('b',3),row('c',4)]
    r=paired_game_bootstrap(rows,[2,3,4,float('inf')],[1,2,3,0],repetitions=100)
    assert r['difference']==1 and r['ci95']==[1,1]
    assert r['games']==2 and r['positions']==3


def test_move_metrics():
    rows=[row('a',1),row('b',2)]
    r=scorecard(BinnedDistribution(torch.zeros(2,1),[0.,10.],'linear'),rows,move_probabilities=torch.tensor([[.7,.2,.1],[.2,.6,.2]]),move_targets=[0,2])
    assert r['groups']['all']['move_top1']==.5
    assert r['groups']['all']['move_top3']==1


def test_scorecard_counts_tiny_probabilities_as_finite_not_unsupported():
    logits=torch.zeros(1,30);logits[:,0]=1000
    from chess_self.timing_distributions import make_distribution
    r=row('a',10.,60.)
    report=scorecard(make_distribution(logits),[r],crps_points=32)
    all_=report['groups']['all']
    assert all_['zero_probability_intervals']==0
    assert all_['interval_nll']>1000
    assert all_['outside_model_support']==0
    json.dumps(report,allow_nan=False)


def test_comparison_labels_nonfinite_subset_and_support_failures():
    from chess_self.timing_metrics import compare_scorecards
    def report(values):
        return {'timing_information':'position_marginal','per_row':[
            {'game_id':str(i),'ply':1,'interval_nll':v,'zero_probability_interval':v is None,
             'crps_seconds':1.,'mean_absolute_error':1.} for i,v in enumerate(values)]}
    result=compare_scorecards(report([2.,None,4.]),report([1.,3.,None]),repetitions=20)
    nll=result['interval_nll']
    assert nll['difference']==1 and nll['positions']==1
    assert nll['excluded_pairs']==2 and nll['input_positions']==3
    assert nll['population']=='common_finite_pairs'
    assert not nll['whole_sample_difference_defined']
    assert nll['left_zero_probability_intervals']==nll['right_zero_probability_intervals']==1
    assert result['crps_seconds']['excluded_pairs']==0
    empty=compare_scorecards(report([None]),report([None]),repetitions=20)['interval_nll']
    assert empty['difference'] is None and empty['excluded_pairs']==1


def test_repetition_reporting_uses_only_training_boards():
    from chess_self.timing_metrics import annotate_training_positions
    training=[{'game_id':'train','fen':'board w - - 0 1'}]
    rows=[{'game_id':'val','fen':'board w - - 99 50'},
          {'game_id':'val','fen':'other b - - 0 1'}]
    annotated=annotate_training_positions(rows,training)
    assert [r['position_seen_in_train'] for r in annotated]==[True,False]
    assert 'position_seen_in_train' not in rows[0]
    with pytest.raises(ValueError,match='overlap'):
        annotate_training_positions(training,training)
