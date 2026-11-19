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
    best = (float("inf"), None)
    for alpha in [1 + x / 10 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512, 1024, 4096, 16384]:
        eps = releases * alpha / (2 * sigma ** 2) + math.log(1 / delta) / (alpha - 1)
        if eps < best[0]:
            best = (eps, alpha)
    return best
