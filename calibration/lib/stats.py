"""Closed-form statistics: OAS covariance shrinkage, LDA (multiclass / binary), isotonic regression (PAV)."""
import numpy as np
import torch


def oas(S, n):
    """Shrink the sample covariance S (p x p, n samples) toward the identity (Oracle Approximating Shrinkage)."""
    p = S.shape[0]; tr = torch.trace(S); tr2 = (S * S).sum(); mu = tr / p
    rho = min(1.0, float(((1 - 2 / p) * tr2 + tr ** 2) / ((n + 1 - 2 / p) * (tr2 - tr ** 2 / p))))
    return (1 - rho) * S + rho * mu * torch.eye(p, device=S.device, dtype=S.dtype)


def lda_multi(X, y, C):
    """C-class LDA (shared covariance, OAS, sample-frequency priors) -> (A, b): discriminant of class c = X @ A[c] + b[c]."""
    X = torch.as_tensor(X, dtype=torch.float64); y = torch.as_tensor(y)
    n = torch.tensor([float((y == c).sum()) for c in range(C)]); mu = torch.stack([X[y == c].mean(0) for c in range(C)])
    Z = torch.cat([X[y == c] - mu[c] for c in range(C)])
    Am = torch.linalg.solve(oas(Z.T @ Z / len(y), len(y)).cpu(), mu.T).T
    return Am, -0.5 * (Am * mu).sum(1) + torch.log(n / n.sum())


def lda2(X, y):
    """Binary LDA (OAS, equal priors) -> (w, b): score = X @ w + b = log p(x|1) / p(x|0)."""
    X = torch.as_tensor(X, dtype=torch.float64); y = torch.as_tensor(y)
    m0, m1 = X[y == 0].mean(0), X[y == 1].mean(0)
    Xc = torch.cat([X[y == 0] - m0, X[y == 1] - m1]); S = oas(Xc.T @ Xc / len(Xc), len(Xc))
    w = torch.linalg.solve(S, m1 - m0)
    return w, -0.5 * (m0 + m1) @ w


def score(X, wb):
    return (torch.as_tensor(X, dtype=torch.float64) @ wb[0] + wb[1]).numpy()


def pav(s, y, w):
    """Weighted isotonic regression: P(y=1) steps non-decreasing in score s -> (right-end score of each step, step probability)."""
    o = np.argsort(s); blk = []
    for x_, y_, w_ in zip(s[o], y[o].astype(float), w[o]):
        blk.append([w_ * y_, w_, x_])
        while len(blk) > 1 and blk[-2][0] / blk[-2][1] >= blk[-1][0] / blk[-1][1]:
            a_ = blk.pop(); blk[-1][0] += a_[0]; blk[-1][1] += a_[1]; blk[-1][2] = a_[2]
    return np.array([b[2] for b in blk]), np.array([b[0] / b[1] for b in blk])
