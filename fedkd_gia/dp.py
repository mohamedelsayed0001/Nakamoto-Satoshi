"""Client-side Gaussian mechanism (the clip-and-noise core of Camel / DP-FedAvg) and its privacy accounting.

The released vector (a client's student update, or a single-step student gradient) is treated as one
flat vector over all shared tensors: it is clipped to L2 norm <= clip, then N(0, (sigma * clip)^2) noise is
added to every coordinate. This protects the whole release of one client (client-level local DP).

Accounting: each release is a Gaussian mechanism with L2 sensitivity `clip` and noise multiplier `sigma`,
whose Renyi DP is alpha / (2 sigma^2). `releases` releases compose additively in RDP, and are converted to
(epsilon, delta)-DP with eps = min_alpha [releases * alpha / (2 sigma^2) + log(1/delta) / (alpha - 1)].
No amplification by subsampling or shuffling is claimed (all clients take part every round, and in this
single-process simulation shuffling only unlinks updates from clients).
"""
import math

import torch


def flat_norm(tensors):
    return math.sqrt(sum(float((t.float() ** 2).sum()) for t in tensors.values()))


def clip_and_noise(tensors, clip, sigma, generator=None):
    """Return (privatized dict, pre-clip L2 norm). tensors: {name: tensor}; noise std = sigma * clip."""
    norm = flat_norm(tensors)
    scale = min(1.0, clip / (norm + 1e-12))
    out = {}
    for k, t in tensors.items():
        t = t.float() * scale
        if sigma > 0:
            t = t + torch.randn(t.shape, generator=generator, device="cpu").to(t.device) * (sigma * clip)
        out[k] = t
    return out, norm


def gaussian_epsilon(sigma, releases, delta=1e-5):
    """(epsilon, best alpha) of `releases` compositions of the Gaussian mechanism with noise multiplier sigma."""
    if sigma <= 0:
        return float("inf"), None
    # f(alpha) = a * alpha + L / (alpha - 1) is minimized exactly at alpha = 1 + sqrt(L / a): eps = a + 2 sqrt(a L).
    a, L = releases / (2 * sigma ** 2), math.log(1 / delta)
    return a + 2 * math.sqrt(a * L), 1 + math.sqrt(L / a)


# ---------------------------------------------------------------------------------------------------------------
# Record-level DP-SGD accounting (Poisson-subsampled Gaussian mechanism).
# Each local step samples every training image independently with probability q, clips each per-image gradient
# to L2 <= C, sums them and adds N(0, (sigma C)^2). For integer Renyi order alpha the RDP of one step is exactly
#   eps(alpha) = 1/(alpha-1) * log sum_k C(alpha,k) (1-q)^(alpha-k) q^k exp((k^2 - k) / (2 sigma^2))
# (Mironov, Talwar, Zhang 2019). Steps compose additively; RDP -> (eps, delta) uses the conversion of
# Balle et al. 2020 (the one Opacus uses). Only integer orders are searched, which can only over-estimate epsilon.
# ---------------------------------------------------------------------------------------------------------------
DPSGD_ORDERS = list(range(2, 256)) + [320, 512, 1024]


def _log_add(a, b):
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m)) if m > -math.inf else -math.inf


def subsampled_gaussian_rdp(q, sigma, alpha):
    if q == 0:
        return 0.0
    if q == 1.0:
        return alpha / (2 * sigma ** 2)
    log_a = -math.inf
    for k in range(alpha + 1):
        log_coef = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        term = log_coef + (alpha - k) * math.log1p(-q) + k * math.log(q) + (k * k - k) / (2 * sigma ** 2)
        log_a = _log_add(log_a, term)
    return log_a / (alpha - 1)


def dpsgd_epsilon(q, sigma, steps, delta=1e-5):
    """(epsilon, best order) after `steps` Poisson-subsampled Gaussian steps with sampling rate q."""
    if sigma <= 0:
        return float("inf"), None
    best = (float("inf"), None)
    for a in DPSGD_ORDERS:
        rdp = steps * subsampled_gaussian_rdp(q, sigma, a)
        eps = rdp + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        if eps < best[0]:
            best = (eps, a)
    return best


def dpsgd_sigma_for_epsilon(target_eps, q, steps, delta=1e-5, tol=1e-4):
    """Smallest noise multiplier whose epsilon after `steps` steps is <= target_eps (bisection)."""
    lo, hi = 0.1, 100.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if dpsgd_epsilon(q, mid, steps, delta)[0] <= target_eps:
            hi = mid
        else:
            lo = mid
    return hi
