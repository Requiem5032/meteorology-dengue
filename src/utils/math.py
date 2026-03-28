import torch


def interp1d(x, xp, fp):
    idx = torch.searchsorted(xp, x) - 1
    idx = idx.clamp(0, len(xp) - 2)

    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]

    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def smooth_safe_divide(numerator, denominator, eps=1e-6):
    numerator_tensor = torch.as_tensor(numerator)
    denominator_tensor = torch.as_tensor(
        denominator,
        dtype=numerator_tensor.dtype,
        device=numerator_tensor.device,
    )
    eps_tensor = torch.as_tensor(
        eps,
        dtype=denominator_tensor.dtype,
        device=denominator_tensor.device,
    )
    scale = torch.hypot(denominator_tensor, eps_tensor)
    return (numerator_tensor / scale) * (denominator_tensor / scale)
