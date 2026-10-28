import pytest
import torch

from chess_self.timing_distributions import (
    BinnedDistribution, BetaTimingDistribution, absolute_edges, make_distribution,
    mixture_distributions, concat_distributions,
)


def uniform(batch=1, upper=10.):
    return BinnedDistribution(torch.zeros(batch, 1, dtype=torch.float64), [0., upper], 'linear')


def test_uniform_cdf_quantiles_and_crps():
    d=uniform(3)
    torch.testing.assert_close(d.cdf(torch.tensor([-1.,5.,20.])), torch.tensor([0.,.5,1.],dtype=torch.float64))
    torch.testing.assert_close(d.mean(), torch.full((3,),5.,dtype=torch.float64))
    torch.testing.assert_close(d.quantile(.25), torch.full((3,),2.5,dtype=torch.float64))
    torch.testing.assert_close(d.crps(torch.tensor([0.,5.,10.]),points=1024),torch.tensor([10/3,10/12,10/3],dtype=torch.float64),atol=2e-5,rtol=0)


def test_partial_log_bin_is_truncated_continuously():
    d=BinnedDistribution(torch.zeros(1,1,dtype=torch.float64), [0.,9.9], 'log')
    assert d.cdf(.9).item()==pytest.approx(.5)
    t=d.truncate(torch.tensor([.9]))
    assert t.removed_mass.item()==pytest.approx(.5)
    assert t.cdf(.9).item()==pytest.approx(1)
    assert t.cdf(.1).item()==pytest.approx(torch.log10(torch.tensor(2.)).item())
    draws=torch.cat([t.sample() for _ in range(100)])
    assert (draws>=0).all() and (draws<.9).all()
    with pytest.raises(ValueError,match='No feasible'):
        d.truncate(0)


def test_interval_gradient_and_support_not_floored():
    logits=torch.zeros(2,30,requires_grad=True)
    d=make_distribution(logits)
    p=d.interval_probability(torch.tensor([1.,180.]),torch.tensor([1.2,181.]))
    assert p[0]>0 and p[1]==0
    (-p[0].log()).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    assert torch.equal(absolute_edges(40)[:31],absolute_edges(30))


def test_fraction_support_does_not_include_increment():
    d=make_distribution(torch.zeros(2,30),'fraction',clock=torch.tensor([2.,20.]))
    torch.testing.assert_close(d.mean(),torch.tensor([1.,10.]))
    assert torch.equal(d.cdf(torch.tensor([2.,20.])),torch.ones(2))
    with pytest.raises(ValueError):make_distribution(torch.zeros(1,30),'fraction',clock=torch.tensor([0.]))


def test_beta_matches_uniform_and_polynomial():
    d=BetaTimingDistribution(torch.tensor([1.,2.],dtype=torch.float64),torch.tensor([1.,3.],dtype=torch.float64),torch.tensor([10.,10.],dtype=torch.float64))
    q=torch.tensor([[0.,.1,.5,.9,1.],[0.,.1,.5,.9,1.]],dtype=torch.float64)
    expected=torch.stack([q[0],6*q[1]**2-8*q[1]**3+3*q[1]**4])
    torch.testing.assert_close(d.cdf(q*10),expected,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(d.cdf(d.quantile(q)),q,atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(d.mean(),torch.tensor([5.,4.],dtype=torch.float64))


def test_beta_interval_has_finite_shape_gradients():
    raw=torch.tensor([[.3,.8]],requires_grad=True)
    d=make_distribution(raw,'beta',clock=torch.tensor([30.]))
    loss=-d.interval_probability(torch.tensor([0.]),torch.tensor([.1])).log().sum()
    loss.backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0


def test_mixture_marginal_uses_policy_weights():
    a,b=uniform(upper=2.),uniform(upper=10.)
    d=mixture_distributions([a,b],torch.tensor([[.25,.75]],dtype=torch.float64))
    assert d.cdf(2).item()==pytest.approx(.4)
    assert d.mean().item()==pytest.approx(4.)
    assert d.cdf(d.quantile(.8)).item()==pytest.approx(.8)
    joined=concat_distributions([a,d.truncate(5)])
    assert joined.cdf(torch.tensor([2.,5.])).tolist()==[1.,1.]
    assert joined.mean().shape==(2,)


def test_seeded_sampling_and_validation():
    d=uniform(4)
    a=d.sample(torch.Generator().manual_seed(9));b=d.sample(torch.Generator().manual_seed(9))
    assert torch.equal(a,b)
    with pytest.raises(ValueError):d.interval_probability(2,1)
    with pytest.raises(ValueError):d.quantile(1.1)
    with pytest.raises(ValueError):BinnedDistribution(torch.zeros(1,2),[0,2,1])


def test_concentrated_bins_keep_small_interval_probability_and_gradients():
    raw = torch.zeros(2, 30, requires_grad=True)
    with torch.no_grad():
        raw[:, 0] = 1000
    d = make_distribution(raw).truncate(torch.tensor([60., 60.]))
    lp = d.log_interval_probability(torch.tensor([9.9, 180.]), torch.tensor([10.1, 181.]))
    assert torch.isfinite(lp[0]) and lp[0] < -1000
    assert torch.isneginf(lp[1])
    assert d.interval_supported(torch.tensor([9.9,180.]),torch.tensor([10.1,181.])).tolist() == [True,False]
    (-lp[0]).backward()
    assert torch.isfinite(raw.grad).all() and raw.grad[0].abs().sum()>0
    assert raw.grad[1].abs().sum()==0
    logits = torch.zeros(1,30); logits[:,0]=30
    assert make_distribution(logits).interval_probability(9.9,10.1).item()>0


def test_beta_extreme_upper_tail_log_probability_is_finite():
    a = torch.tensor([1.],dtype=torch.float64,requires_grad=True)
    b = torch.tensor([1000.],dtype=torch.float64,requires_grad=True)
    d=BetaTimingDistribution(a,b,torch.tensor([10.],dtype=torch.float64))
    lp=d.log_interval_probability(9.,9.1)
    expected=1000*torch.log(torch.tensor(.1,dtype=torch.float64))+torch.log1p(-torch.tensor(.9,dtype=torch.float64)**1000)
    torch.testing.assert_close(lp[0],expected,atol=1e-8,rtol=0)
    (-lp.sum()).backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_common_bins_mixture_fast_path_matches_conditional_marginal():
    from chess_self.timing_distributions import MixtureDistribution, TruncatedDistribution
    a=BinnedDistribution(torch.tensor([[2.,-2.]],dtype=torch.float64),[0.,2.,10.],'linear')
    b=BinnedDistribution(torch.tensor([[-2.,2.]],dtype=torch.float64),[0.,2.,10.],'linear')
    weights=torch.tensor([[.2,.8]],dtype=torch.float64)
    fast=mixture_distributions([a,b],weights)
    assert isinstance(fast,BinnedDistribution)
    slow=MixtureDistribution([a,b],weights)
    query=torch.tensor([[0.,1.,2.,4.,10.]],dtype=torch.float64)
    torch.testing.assert_close(fast.cdf(query),slow.cdf(query))
    at,bt=a.truncate(3.),b.truncate(3.)
    fast=mixture_distributions([at,bt],weights)
    assert isinstance(fast,TruncatedDistribution) and isinstance(fast.base,BinnedDistribution)
    slow=MixtureDistribution([at,bt],weights)
    torch.testing.assert_close(fast.cdf(query),slow.cdf(query),atol=1e-12,rtol=0)
    torch.testing.assert_close(fast.log_interval_probability(.2,2.4),slow.log_interval_probability(.2,2.4))
    torch.testing.assert_close(fast.mean(),slow.mean())
    torch.testing.assert_close(fast.removed_mass,.2*at.removed_mass+.8*bt.removed_mass)


def test_rare_truncation_keeps_conditional_quantile_and_mean():
    d=BinnedDistribution(torch.tensor([[-1000.,0.]],dtype=torch.float64),[0.,2.,10.],'linear').truncate(1.)
    assert d.log_retained_mass.item() < -1000
    assert d.quantile(.5).item()==pytest.approx(.5)
    assert d.mean().item()==pytest.approx(.5)
    assert d.cdf(.25).item()==pytest.approx(.25)
