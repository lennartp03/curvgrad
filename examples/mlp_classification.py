# %%
import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles
from two_moons import FIGURES, generate_two_moons, grid

from curvgrad import autograd
from curvgrad.curvature.ggn import flatten, ggn_matrix, jacobian, unflatten
from curvgrad.curvature.laplace import laplace_posterior_precision, margin_moments, probit
from curvgrad.nn.layers import MLP, Linear
from curvgrad.nn.loss import cross_entropy_hvp, cross_entropy_loss
from curvgrad.nn.module import Module
from curvgrad.tensor import Tensor

plt.rcParams.update(bundles.tue_ai_beamer())
plt.rcParams.update({"figure.dpi": 150})

X, y = generate_two_moons(200, random_state=0)
Y = np.eye(2)[y.astype(int)]
X_test, y_test = generate_two_moons(1000, random_state=1)
n_data, prior_prec = X.shape[0], 1.0


def fit_map(model: Module, steps: int = 20000, lr: float = 0.5) -> float:
    """Gradient descent on E / N = L + tau / (2N) |theta|^2. Returns the final |grad E|."""
    params = model.parameters()
    for _ in range(steps):
        g = autograd.vjp(cross_entropy_loss(model(Tensor(X)), Y))
        for p in params:
            p.data -= lr * (g[p] + prior_prec / n_data * p.data)
    g = autograd.vjp(cross_entropy_loss(model(Tensor(X)), Y))
    theta = flatten([p.data for p in params])
    return float(np.linalg.norm(n_data * flatten([g[p] for p in params]) + prior_prec * theta))


def laplace_chol(model: Module) -> np.ndarray:
    """Cholesky factor of the posterior precision N G + tau I at the current parameters."""
    out = model(Tensor(X))
    ggn = ggn_matrix(out, model.parameters(), lambda u: cross_entropy_hvp(out.data, u))
    return np.linalg.cholesky(laplace_posterior_precision(ggn, n_data, prior_prec))


def predictive(
    model: Module, chol: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MAP and probit Laplace probability of class 1, plus the margin standard deviation."""
    out = model(Tensor(points))
    mean, var = margin_moments(jacobian(out, model.parameters()), out.data, chol)
    return probit(mean, np.zeros_like(var)), probit(mean, var), np.sqrt(var)


# %%
# MAP and Laplace for a linear model and a relu MLP, with the same code path for both
np.random.seed(0)
models: dict[str, Module] = {"Linear": Linear(2, 2), "MLP": MLP([2, 16, 2])}
chols = {}
for name, model in models.items():
    grad_norm = fit_map(model)
    chols[name] = laplace_chol(model)
    P = sum(p.data.size for p in model.parameters())
    print(f"{name}: P = {P}, |grad E| = {grad_norm:.1e}")

# %%
# Test set and far away points. Far from the data an honest model should approach p = 0.5.
angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
far = X.mean(axis=0) + 10 * np.column_stack([np.cos(angles), np.sin(angles)])
for name, model in models.items():
    p_map, p_lap, _ = predictive(model, chols[name], X_test)
    f_map, f_lap, _ = predictive(model, chols[name], far)
    for kind, p1, pf in [("MAP", p_map, f_map), ("Laplace", p_lap, f_lap)]:
        p1 = np.clip(p1, 1e-12, 1 - 1e-12)
        nll = -np.mean(y_test * np.log(p1) + (1 - y_test) * np.log(1 - p1))
        acc = np.mean((p1 > 0.5) == y_test)
        conf = np.mean(np.maximum(pf, 1 - pf))
        print(
            f"{name:6s} {kind:7s} test acc {acc:.3f}  test NLL {nll:.3f}  confidence at r = 10 {conf:.3f}"
        )

# %%
xx, yy, points = grid(X)
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(9, 5.5))
for row, (name, model) in zip(axes, models.items(), strict=True):
    p_map, p_lap, std = predictive(model, chols[name], points)
    panels = [
        (p_map, "MAP", "RdBu_r"),
        (p_lap, "Laplace", "RdBu_r"),
        (std, "margin std", "viridis"),
    ]
    for ax, (z, title, cmap) in zip(row, panels, strict=True):
        cs = ax.contourf(xx, yy, z.reshape(xx.shape), levels=11, cmap=cmap)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", s=3, edgecolors="none")
        ax.set_title(f"{name}, {title}")
        fig.colorbar(cs, ax=ax)
fig.savefig(FIGURES / "mlp_vs_linear.png", bbox_inches="tight")
plt.show()


# %%
# GGN against the full Hessian. The relu Hessian is undefined at the kinks and a MAP usually has a
# preactivation almost exactly at one, so this comparison uses smooth sigmoid hidden units.
def fd_hessian(model: Module, h: float = 1e-5) -> np.ndarray:
    """Hessian of the mean loss by central differences of the exact gradient, one column per parameter."""
    params = model.parameters()
    like = [p.data.copy() for p in params]
    theta = flatten(like)

    def grad(t: np.ndarray) -> np.ndarray:
        for p, piece in zip(params, unflatten(t, like), strict=True):
            p.data[...] = piece
        g = autograd.vjp(cross_entropy_loss(model(Tensor(X)), Y))
        return flatten([g[p] for p in params])

    H = np.stack(
        [(grad(theta + h * e) - grad(theta - h * e)) / (2 * h) for e in np.eye(theta.size)], 1
    )
    grad(theta)  # restore the parameters
    return H


def curvature(model: Module) -> tuple[np.ndarray, np.ndarray]:
    """Dense GGN and finite difference Hessian of the mean loss at the current parameters."""
    out = model(Tensor(X))
    G = ggn_matrix(out, model.parameters(), lambda u: cross_entropy_hvp(out.data, u))
    return G, fd_hessian(model)


np.random.seed(0)
smooth = MLP([2, 16, 2], activation="sigmoid")
stages = {"initialization": curvature(smooth)}
fit_map(smooth, lr=2.0)
stages["MAP"] = curvature(smooth)

blocks = np.cumsum([p.data.size for p in smooth.parameters()])[:-1]
fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
for row, (stage, (G, H)) in zip(axes, stages.items(), strict=True):
    eig_G, eig_H = np.linalg.eigvalsh(G), np.linalg.eigvalsh((H + H.T) / 2)
    rel = np.linalg.norm(H - G) / np.linalg.norm(H)
    print(
        f"{stage}: |H - G| / |H| = {rel:.3f}, negative Hessian eigenvalues {(eig_H < -1e-8).sum()}"
    )
    # A Laplace precision built from the Hessian is only valid if this is positive
    print(f"  smallest eigenvalue of N H + tau I: {n_data * eig_H.min() + prior_prec:.2f}")
    vmax = np.abs(H).max()
    for ax, (M, title) in zip(
        row[:3], [(G, "GGN"), (H, "Hessian"), (H - G, "Hessian - GGN")], strict=True
    ):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for edge in blocks:  # borders of W1, b1, W2, b2
            ax.axhline(edge - 0.5, color="gray", linewidth=0.5)
            ax.axvline(edge - 0.5, color="gray", linewidth=0.5)
        ax.set_title(f"{title}, {stage}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=row[:3].tolist(), shrink=0.8)
    ax = row[3]
    ax.plot(eig_H[::-1], label="Hessian")
    ax.plot(eig_G[::-1], label="GGN", linestyle="--")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_title("eigenvalues, largest first")
    ax.legend()
fig.savefig(FIGURES / "mlp_ggn_vs_hessian.png", bbox_inches="tight")
plt.show()
