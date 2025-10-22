import torch


def torch_interp(t_eval, t, y):
    t_eval = t_eval.float()
    t = t.float()
    y = y.float()

    t_eval_clamped = torch.clamp(t_eval, t[0], t[-1])

    idxs = torch.searchsorted(t, t_eval_clamped, right=True)
    idxs = torch.clamp(idxs, 1, t.numel() - 1)
    x0 = t[idxs - 1]
    x1 = t[idxs]
    y0 = y[idxs - 1]
    y1 = y[idxs]
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (t_eval_clamped - x0)
