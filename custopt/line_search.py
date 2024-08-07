import torch

def _armijo(f, x, gx, dx, t, alpha=0.1, beta=0.5):
    f0 = f(x, 0, dx)
    f1 = f(x, t, dx)
    while f1 > f0 + alpha * t * torch.abs(gx.vdot(dx)):
        t *= beta
        f1 = f(x, t, dx)
    return t