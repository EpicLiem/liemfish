"""Duration laws in seconds; absolute bins are uniform in log(t + 0.1)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def absolute_edges(bins=30, *, device=None, dtype=torch.float32):
    edges = 10 ** (torch.arange(bins + 1, device=device, dtype=dtype) / 10 - 1) - .1
    edges[0] = 0
    return edges


def _batch(value, reference):
    value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if value.ndim == 0:
        value = value.expand(reference.shape[0])
    if value.shape[0] != reference.shape[0] or value.ndim not in (1, 2):
        raise ValueError("Query must be scalar, [batch], or [batch, queries]")
    return value


def _log_weights(weights):
    positive = weights > 0
    return torch.where(positive, torch.where(positive, weights, 1.).double().log(), -torch.inf)


def _expand(value, query):
    return value.unsqueeze(-1) if query.ndim == 2 else value


class DurationDistribution:
    """Batched continuous distributions; no implicit observation rounding or clipping."""

    def interval_probability(self, low, high):
        return self.log_interval_probability(low, high).exp()

    def interval_supported(self, low, high):
        low, high = _batch(low, self.reference), _batch(high, self.reference)
        if torch.any(high < low):
            raise ValueError("Observation interval is reversed")
        return torch.minimum(high, _expand(self.upper, high)) > torch.maximum(low, _expand(self.lower, low))

    def log_interval_probability(self, low, high):
        raise NotImplementedError

    def sample(self, generator=None):
        q = torch.rand(self.reference.shape[0], dtype=self.reference.dtype,
                       device=self.reference.device, generator=generator)
        result = self.quantile(q)
        return torch.maximum(self.lower, torch.minimum(result, torch.nextafter(self.upper, self.lower)))

    def truncate(self, budget):
        return TruncatedDistribution(self, budget)

    def crps(self, observed, points=256):
        """Midpoint quantile quadrature of CRPS in seconds (increase points to check accuracy)."""
        y = _batch(observed, self.reference)
        if y.ndim != 1 or points < 2:
            raise ValueError("CRPS expects one observation per row and at least two points")
        q = (torch.arange(points, device=y.device, dtype=y.dtype) + .5) / points
        q = q.unsqueeze(0).expand(len(y), -1)
        error = y[:, None] - self.quantile(q)
        return 2 * (error * (q - (error < 0).to(q.dtype))).mean(-1)


class BinnedDistribution(DurationDistribution):
    def __init__(self, logits, edges, law="log", *, _clipped=False):
        if logits.ndim != 2 or law not in {"log", "linear"}:
            raise ValueError("Expected [batch, bins] logits and log or linear law")
        if not _clipped and not torch.isfinite(logits).all():
            raise ValueError("Timing logits must be finite")
        edges = torch.as_tensor(edges, dtype=logits.dtype, device=logits.device)
        if edges.ndim == 1:
            edges = edges.unsqueeze(0).expand(len(logits), -1)
        if edges.shape != (len(logits), logits.shape[1] + 1):
            raise ValueError("Edges must have one more column than logits")
        invalid_width = (edges[:, 1:] < edges[:, :-1]) if _clipped else (edges[:, 1:] <= edges[:, :-1])
        if not torch.isfinite(edges).all() or (edges < 0).any() or invalid_width.any():
            raise ValueError("Edges must be finite, nonnegative and strictly increasing")
        self.reference = logits[:, 0]
        self.logits, self.probabilities, self.edges, self.law = logits, logits.softmax(-1), edges, law
        self.log_probabilities = logits.double().log_softmax(-1)
        self.lower, self.upper = edges[:, 0], edges[:, -1]

    def _transform(self, x):
        return (x.clamp_min(0) + .1).log() if self.law == "log" else x

    def log_interval_probability(self, low, high):
        supported = self.interval_supported(low, high)
        low, high = _batch(low, self.reference).double(), _batch(high, self.reference).double()
        a, b = self.edges[:, :-1].double(), self.edges[:, 1:].double()
        logp = self.log_probabilities
        if low.ndim == 2:
            a, b, logp = a[:, None], b[:, None], logp[:, None]
        left, right = torch.maximum(low[..., None], a), torch.minimum(high[..., None], b)
        overlap = (right > left) & (b > a)
        width = self._transform(right) - self._transform(left)
        full = self._transform(b) - self._transform(a)
        logfraction = torch.where(overlap, width, 1.).log() - torch.where(full > 0, full, 1.).log()
        terms = torch.where(overlap, logp + logfraction, -torch.inf)
        # Avoid the undefined gradient of logsumexp([-inf,...]) on excluded labels.
        terms = torch.where(supported[..., None], terms, torch.zeros_like(terms))
        result = torch.logsumexp(terms, -1)
        return torch.where(supported, result, -torch.inf)

    def cdf(self, value):
        t = _batch(value, self.reference)
        a, b = self._transform(self.edges[:, :-1]), self._transform(self.edges[:, 1:])
        p = self.probabilities
        if t.ndim == 2:
            a, b, p = a[:, None], b[:, None], p[:, None]
        f = ((self._transform(t)[..., None] - a) / (b - a).clamp_min(torch.finfo(a.dtype).tiny)).clamp(0, 1)
        return (p * f).sum(-1).clamp(0, 1)

    def quantile(self, value):
        q = _batch(value, self.reference)
        if ((q < 0) | (q > 1)).any():
            raise ValueError("Quantiles must be in [0,1]")
        squeeze = q.ndim == 1
        q = q[:, None] if squeeze else q
        cumulative = self.log_probabilities.exp().cumsum(-1)
        # The final cumulative probability is exactly one for endpoint queries.
        index = (q[:, :, None] > cumulative[:, None]).sum(-1).clamp_max(self.probabilities.shape[1] - 1)
        prior = F.pad(cumulative, (1, 0))[:, :-1].gather(1, index)
        p = self.log_probabilities.exp().gather(1, index)
        fraction = ((q - prior) / p.clamp_min(torch.finfo(p.dtype).tiny)).clamp(0, 1)
        left = self._transform(self.edges[:, :-1]).gather(1, index)
        right = self._transform(self.edges[:, 1:]).gather(1, index)
        result = left + fraction * (right - left)
        if self.law == "log":
            result = result.exp() - .1
        result = torch.where(q == 0, self.lower[:, None], result)
        result = torch.where(q == 1, self.upper[:, None], result)
        return result[:, 0] if squeeze else result

    def mean(self):
        a, b = self.edges[:, :-1], self.edges[:, 1:]
        centers = (b - a) / ((b + .1).log() - (a + .1).log()).clamp_min(torch.finfo(a.dtype).tiny) - .1 if self.law == "log" else (a + b) / 2
        return (self.probabilities * centers).sum(-1)


def _beta_fraction(a, b, x):
    """Lentz continued fraction for regularized incomplete beta; differentiable in a,b."""
    tiny = torch.finfo(x.dtype).tiny * 100
    def nonzero(v):
        return torch.where(v.abs() < tiny, torch.where(v < 0, -tiny, tiny), v)
    qab, qap, qam = a + b, a + 1, a - 1
    c = torch.ones_like(x)
    d = 1 / nonzero(1 - qab * x / qap)
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 / nonzero(1 + aa * d)
        c = nonzero(1 + aa / c)
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 / nonzero(1 + aa * d)
        c = nonzero(1 + aa / c)
        delta = d * c
        h = h * delta
        if bool(((delta - 1).abs() < 1e-10).all().detach()):
            break
    else:
        raise ArithmeticError("Incomplete beta did not converge; inspect shape parameters")
    return h


def regularized_beta(x, alpha, beta):
    # Double precision prevents endpoint and tail cancellation in interval scores.
    dtype = x.dtype
    x, a, b = x.double(), alpha.double(), beta.double()
    interior = x.clamp(torch.finfo(torch.float64).eps, 1 - torch.finfo(torch.float64).eps)
    direct = interior < (a + 1) / (a + b + 2)
    aa = torch.where(direct, a, b)
    bb = torch.where(direct, b, a)
    xx = torch.where(direct, interior, 1 - interior)
    factor = (torch.lgamma(aa + bb) - torch.lgamma(aa) - torch.lgamma(bb)
              + aa * xx.log() + bb * torch.log1p(-xx)).exp()
    result = factor * _beta_fraction(aa, bb, xx) / aa
    result = torch.where(direct, result, 1 - result)
    result = torch.where(x <= 0, 0., torch.where(x >= 1, 1., result))
    return result.clamp(0, 1).to(dtype)


def _log_complement(value):
    negative = value < 0
    safe = torch.where(negative, value, -torch.ones_like(value))
    return torch.where(negative, (-torch.expm1(safe)).log(), -torch.inf)


def _log_difference(large, small):
    finite = torch.isfinite(large)
    delta = torch.where(finite, small - large, -torch.ones_like(large))
    return torch.where(finite, large + _log_complement(delta), -torch.inf)


def log_regularized_beta(x, alpha, beta):
    x, a, b = x.double(), alpha.double(), beta.double()
    interior = x.clamp(torch.finfo(torch.float64).tiny, 1 - torch.finfo(torch.float64).eps)
    direct = interior < (a + 1) / (a + b + 2)
    aa, bb = torch.where(direct, a, b), torch.where(direct, b, a)
    xx = torch.where(direct, interior, 1 - interior)
    logtail = (torch.lgamma(aa + bb) - torch.lgamma(aa) - torch.lgamma(bb)
               + aa * xx.log() + bb * torch.log1p(-xx)
               + _beta_fraction(aa, bb, xx).log() - aa.log()).clamp_max(0)
    result = torch.where(direct, logtail, _log_complement(logtail))
    return torch.where(x <= 0, -torch.inf, torch.where(x >= 1, 0., result))


class BetaTimingDistribution(DurationDistribution):
    def __init__(self, alpha, beta, clock):
        alpha, beta, clock = torch.broadcast_tensors(alpha, beta, torch.as_tensor(clock, device=alpha.device, dtype=alpha.dtype))
        if alpha.ndim != 1 or not torch.isfinite(torch.stack((alpha, beta, clock))).all():
            raise ValueError("Beta parameters and clocks must be finite batch vectors")
        if ((alpha <= 0) | (beta <= 0) | (clock <= 0)).any():
            raise ValueError("Beta shapes and pre-move clock must be positive")
        self.alpha, self.beta, self.clock, self.reference = alpha, beta, clock, alpha
        self.lower, self.upper = torch.zeros_like(clock), clock

    def cdf(self, value):
        value = _batch(value, self.reference)
        return regularized_beta(value / _expand(self.clock, value), _expand(self.alpha, value), _expand(self.beta, value))

    def log_interval_probability(self, low, high):
        supported = self.interval_supported(low, high)
        low, high = _batch(low, self.reference).double(), _batch(high, self.reference).double()
        a, b, clock = (_expand(t.double(), low) for t in (self.alpha, self.beta, self.clock))
        lo, hi = (low / clock).clamp(0, 1), (high / clock).clamp(0, 1)
        # Subtract survival probabilities in the upper tail, CDFs in the lower tail.
        upper_tail = lo >= a / (a + b)
        first = torch.where(upper_tail, 1-lo, hi)
        second = torch.where(upper_tail, 1-hi, lo)
        aa, bb = torch.where(upper_tail, b, a), torch.where(upper_tail, a, b)
        # Excluded intervals still need benign intermediate derivatives.
        first = torch.where(supported, first, .75)
        second = torch.where(supported, second, .25)
        result = _log_difference(log_regularized_beta(first, aa, bb), log_regularized_beta(second, aa, bb))
        # Very narrow interior intervals can round two log-CDFs to the same value.
        # Local Gaussian quadrature avoids their subtraction; endpoint singularities
        # are handled above by the incomplete beta rather than this fallback.
        fallback = supported & ~torch.isfinite(result)
        if bool(fallback.any().detach()):
            import numpy as np
            nodes, weights = np.polynomial.legendre.leggauss(32)
            nodes = torch.as_tensor(nodes, device=lo.device, dtype=lo.dtype)
            weights = torch.as_tensor(weights, device=lo.device, dtype=lo.dtype)
            width = torch.where(supported, hi-lo, torch.ones_like(lo))
            x = lo[..., None] + width[..., None] * (nodes+1)/2
            x = x.clamp(torch.finfo(x.dtype).tiny, 1-torch.finfo(x.dtype).eps)
            logpdf = ((a[..., None]-1)*x.log() + (b[..., None]-1)*torch.log1p(-x)
                      + (torch.lgamma(a+b)-torch.lgamma(a)-torch.lgamma(b))[..., None])
            local = torch.logsumexp(logpdf + weights.log(), -1) + (width/2).log()
            result = torch.where(fallback, local, result)
        return torch.where(supported, result, -torch.inf)

    def quantile(self, value):
        q = _batch(value, self.reference)
        if ((q < 0) | (q > 1)).any():
            raise ValueError("Quantiles must be in [0,1]")
        lo, hi = torch.zeros_like(q), torch.ones_like(q)
        for _ in range(42):
            mid = (lo + hi) / 2
            below = regularized_beta(mid, _expand(self.alpha, q), _expand(self.beta, q)) < q
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
        return torch.where(q == 0, 0., torch.where(q == 1, 1., (lo + hi) / 2)) * _expand(self.clock, q)

    def mean(self):
        return self.clock * self.alpha / (self.alpha + self.beta)


class TruncatedDistribution(DurationDistribution):
    def __init__(self, base, budget):
        budget = _batch(budget, base.reference)
        if budget.ndim != 1 or not torch.isfinite(budget).all() or (budget <= base.lower).any():
            raise ValueError("No feasible duration: budget must exceed lower support")
        self.base, self.reference = base, base.reference
        self.lower, self.upper = base.lower, torch.minimum(base.upper, budget)
        self.log_retained_mass = base.log_interval_probability(self.lower, self.upper)
        self.retained_mass = self.log_retained_mass.exp()
        if not torch.isfinite(self.log_retained_mass).all():
            raise ValueError("No probability in feasible duration support")
        self.removed_mass = 1 - self.retained_mass
        self._binned = None
        if isinstance(base, BinnedDistribution):
            clipped = torch.minimum(base.edges, self.upper[:, None])
            logmass = base.log_interval_probability(clipped[:, :-1], clipped[:, 1:])
            self._binned = BinnedDistribution(logmass, clipped, base.law, _clipped=True)

    def log_interval_probability(self, low, high):
        supported = self.interval_supported(low, high)
        low, high = _batch(low, self.reference), _batch(high, self.reference)
        upper = _expand(self.upper, high)
        left = torch.maximum(low, _expand(self.lower, low)).minimum(upper)
        right = high.minimum(upper).maximum(left)
        logp = self.base.log_interval_probability(left, right) - _expand(self.log_retained_mass, high)
        return torch.where(supported, logp, -torch.inf)

    def cdf(self, value):
        value = _batch(value, self.reference)
        lo = _expand(self.lower, value).expand_as(value)
        return self.log_interval_probability(lo, value.maximum(lo)).exp().clamp(0, 1)

    def quantile(self, value):
        value = _batch(value, self.reference)
        if ((value < 0) | (value > 1)).any():
            raise ValueError("Quantiles must be in [0,1]")
        if self._binned is not None:
            return self._binned.quantile(value)
        if (self.log_retained_mass < -80).any():
            lo, hi = _expand(self.lower, value).expand_as(value), _expand(self.upper, value).expand_as(value)
            for _ in range(48):
                mid = (lo + hi) / 2
                below = self.cdf(mid) < value
                lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
            return (lo + hi) / 2
        return torch.minimum(self.base.quantile(value * _expand(self.retained_mass, value)), _expand(self.upper, value))

    def mean(self):
        if self._binned is not None:
            return self._binned.mean()
        # Quantile quadrature also works for Beta and mixtures; 512 midpoint samples.
        q = (torch.arange(512, device=self.reference.device, dtype=self.reference.dtype) + .5) / 512
        return self.quantile(q[None].expand(len(self.reference), -1)).mean(-1)


class MixtureDistribution(DurationDistribution):
    def __init__(self, components, weights):
        if not components or weights.ndim != 2 or weights.shape != (len(components[0].reference), len(components)):
            raise ValueError("Mixture requires [batch, components] policy weights")
        if not torch.isfinite(weights).all() or (weights < 0).any() or (weights.sum(-1) <= 0).any():
            raise ValueError("Invalid mixture weights")
        self.components, self.weights = components, weights / weights.sum(-1, keepdim=True)
        self.reference = components[0].reference
        self.lower = torch.stack([c.lower for c in components]).min(0).values
        self.upper = torch.stack([c.upper for c in components]).max(0).values

    def interval_supported(self, low, high):
        low = _batch(low, self.reference)
        supported = torch.stack([c.interval_supported(low, high) for c in self.components], -1)
        weights = self.weights[:, None] if low.ndim == 2 else self.weights
        return (supported & (weights > 0)).any(-1)

    def cdf(self, value):
        value = _batch(value, self.reference)
        result = torch.stack([c.cdf(value) for c in self.components], -1)
        weights = self.weights[:, None] if value.ndim == 2 else self.weights
        return (weights * result).sum(-1).clamp(0, 1)

    def log_interval_probability(self, low, high):
        supported = self.interval_supported(low, high)
        low = _batch(low, self.reference)
        logp = torch.stack([c.log_interval_probability(low, high) for c in self.components], -1)
        weights = self.weights[:, None] if low.ndim == 2 else self.weights
        terms = logp + _log_weights(weights)
        has_mass = torch.isfinite(terms).any(-1)
        terms = torch.where(has_mass[..., None], terms, torch.zeros_like(terms))
        return torch.where(supported & has_mass, torch.logsumexp(terms, -1), -torch.inf)

    def mean(self):
        return (self.weights * torch.stack([c.mean() for c in self.components], -1)).sum(-1)

    def quantile(self, value):
        q = _batch(value, self.reference)
        if ((q < 0) | (q > 1)).any():
            raise ValueError("Quantiles must be in [0,1]")
        lo, hi = torch.broadcast_tensors(_expand(self.lower, q), q)[0], torch.broadcast_tensors(_expand(self.upper, q), q)[0]
        for _ in range(42):
            mid = (lo + hi) / 2
            below = self.cdf(mid) < q
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
        return torch.where(q == 0, _expand(self.lower, q), torch.where(q == 1, _expand(self.upper, q), (lo + hi) / 2))


def mixture_distributions(distributions, weights):
    mixture = MixtureDistribution(distributions, weights)
    truncated = all(isinstance(d, TruncatedDistribution) and isinstance(d.base, BinnedDistribution) for d in distributions)
    bases = [d.base for d in distributions] if truncated else distributions
    shared_bins = isinstance(bases[0], BinnedDistribution) and all(isinstance(d, BinnedDistribution) and d.law == bases[0].law and torch.equal(d.edges, bases[0].edges) for d in bases)
    shared_budget = not truncated or all(torch.equal(d.upper, distributions[0].upper) for d in distributions)
    if shared_bins and shared_budget:
        logp = torch.stack([d.log_probabilities for d in bases], 1)
        if truncated:
            logp = logp - torch.stack([d.log_retained_mass for d in distributions], 1)[..., None]
        logits = torch.logsumexp(logp + _log_weights(mixture.weights)[..., None], 1)
        combined = BinnedDistribution(logits, bases[0].edges, bases[0].law)
        if truncated:
            combined = combined.truncate(distributions[0].upper)
            combined.removed_mass = (mixture.weights * torch.stack([d.removed_mass for d in distributions], 1)).sum(1)
        return combined
    return mixture


def make_distribution(parameters, family="absolute", clock=None):
    if family in {"absolute", "extended"}:
        bins = 30 if family == "absolute" else 40
        if parameters.shape[-1] != bins:
            raise ValueError(f"{family} requires {bins} output logits")
        return BinnedDistribution(parameters, absolute_edges(bins, device=parameters.device, dtype=parameters.dtype))
    if family == "fraction":
        if clock is None:
            raise ValueError("Fraction output requires pre-move clock")
        clock = _batch(clock, parameters[:, 0])
        if clock.ndim != 1 or (clock <= 0).any():
            raise ValueError("Fraction output requires positive batch clocks")
        edges = clock[:, None] * torch.linspace(0, 1, parameters.shape[-1] + 1, device=parameters.device, dtype=parameters.dtype)
        return BinnedDistribution(parameters, edges, law="linear")
    if family == "beta":
        if parameters.shape[-1] != 2 or clock is None:
            raise ValueError("Beta requires two raw parameters and pre-move clock")
        shapes = F.softplus(parameters) + .05
        return BetaTimingDistribution(shapes[:, 0], shapes[:, 1], clock)
    raise ValueError(f"Unknown timing family: {family}")


class ConcatenatedDistribution(DurationDistribution):
    def __init__(self, batches):
        if not batches:
            raise ValueError("No distribution batches")
        self.batches = list(batches)
        self.reference = torch.cat([b.reference for b in batches])
        self.lower = torch.cat([b.lower for b in batches])
        self.upper = torch.cat([b.upper for b in batches])
        if all(hasattr(b, 'removed_mass') for b in batches):
            self.removed_mass = torch.cat([b.removed_mass for b in batches])

    def _apply(self, method, value):
        value = _batch(value, self.reference)
        chunks = value.split([len(b.reference) for b in self.batches])
        return torch.cat([getattr(b, method)(chunk) for b, chunk in zip(self.batches, chunks)])

    def cdf(self, value):
        return self._apply('cdf', value)

    def log_interval_probability(self, low, high):
        low, high = _batch(low, self.reference), _batch(high, self.reference)
        lengths = [len(b.reference) for b in self.batches]
        return torch.cat([b.log_interval_probability(a, z) for b, a, z in zip(self.batches, low.split(lengths), high.split(lengths))])

    def interval_supported(self, low, high):
        low, high = _batch(low, self.reference), _batch(high, self.reference)
        lengths = [len(b.reference) for b in self.batches]
        return torch.cat([b.interval_supported(a, z) for b, a, z in zip(self.batches, low.split(lengths), high.split(lengths))])

    def quantile(self, value):
        return self._apply('quantile', value)

    def mean(self):
        return torch.cat([b.mean() for b in self.batches])


def concat_distributions(batches):
    return ConcatenatedDistribution(batches)
