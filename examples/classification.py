# %%
import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles
from two_moons import FIGURES, generate_two_moons, grid

from curvgrad import autograd
from curvgrad.curvature.ggn import flatten, ggn_matrix, jacobian, unflatten
from curvgrad.curvature.laplace import (
    laplace_posterior_precision,
    log_marginal_likelihood,
    newton_step,
    sample,
)
from curvgrad.nn.layers import Linear
from curvgrad.nn.loss import cross_entropy_hvp, cross_entropy_loss
from curvgrad.tensor import Tensor

plt.rcParams.update(bundles.tue_ai_beamer())
plt.rcParams.update({"figure.dpi": 150})

X, y = generate_two_moons(200, random_state=0)
Y = np.eye(2)[y.astype(int)]  # one hot labels, shape (N, 2)


# %%
def plot_two_moons(model=None):
    """Scatter both classes. With a model, also draw its class 1 probability and the boundary."""
    _, ax = plt.subplots()

    if model is not None:
        lo, hi = X.min(axis=0) - 0.5, X.max(axis=0) + 0.5
        xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
        logits = model(Tensor(np.column_stack([xx.ravel(), yy.ravel()]))).data
        margin = np.clip(logits[:, 0] - logits[:, 1], -50, 50)
        p1 = (1 / (1 + np.exp(margin))).reshape(xx.shape)  # softmax probability of class 1
        ax.contourf(xx, yy, p1, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.3)
        ax.contour(xx, yy, p1, levels=[0.5], colors="black", linewidths=1)

    ax.scatter(X[y == 0, 0], X[y == 0, 1], alpha=0.7)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], alpha=0.7)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


plot_two_moons()

# %%
np.random.seed(0)
epochs = 500
lr = 0.5
history: list[float] = []
lm = Linear(2, 2)

for i in range(epochs):
    loss = cross_entropy_loss(lm(Tensor(X)), Y)
    g = autograd.vjp(loss)
    for p in lm.parameters():
        p.data -= lr * g[p]
    history.append(float(loss.data))
    if i % 100 == 0:
        print(f"Loss at iter {i}: {history[-1]:.4f}")

accuracy = (lm(Tensor(X)).data.argmax(axis=1) == y).mean()
print(f"Train accuracy: {accuracy:.3f}")

# %%
plot_two_moons(lm)

# %%
out = lm(Tensor(X))
ggn = ggn_matrix(out, lm.parameters(), lambda u: cross_entropy_hvp(out.data, u))
labels = ["W00", "W01", "W10", "W11", "b0", "b1"]

fig, (ax1, ax2) = plt.subplots(1, 2)
im = ax1.imshow(ggn, cmap="RdBu_r", vmin=-np.abs(ggn).max(), vmax=np.abs(ggn).max())
ax1.set_xticks(range(6), labels)
ax1.set_yticks(range(6), labels)
ax1.set_title("GGN")
fig.colorbar(im, ax=ax1)
ax2.bar(range(6), np.clip(np.linalg.eigvalsh(ggn), 1e-16, None))
ax2.set_yscale("log")
ax2.set_title("eigenvalues")
fig.savefig(FIGURES / "linear_ggn.png", bbox_inches="tight")
plt.show()

# %%
# MAP with a Gaussian prior via Newton, which equals IRLS for this model
n_data = X.shape[0]
prior_prec = 1.0
params = lm.parameters()
like = [p.data for p in params]
for p in params:
    p.data[...] = 0.0

for i in range(20):
    out = lm(Tensor(X))  # fresh graph after every update
    g = autograd.vjp(cross_entropy_loss(out, Y))
    theta = flatten([p.data for p in params])
    grad_mean = flatten([g[p] for p in params])
    grad_E = n_data * grad_mean + prior_prec * theta
    print(f"Newton step {i}: |grad E| = {np.linalg.norm(grad_E):.1e}")
    if np.linalg.norm(grad_E) < 1e-10:
        break
    ggn = ggn_matrix(out, params, lambda u: cross_entropy_hvp(out.data, u))
    theta = newton_step(theta, grad_mean, ggn, n_data, prior_prec)
    for p, piece in zip(params, unflatten(theta, like), strict=True):
        p.data[...] = piece

theta_map = flatten([p.data for p in params])
print("theta_map", np.round(theta_map, 3))

# %%
# Laplace posterior N(theta_map, Lambda^-1) and samples from it
out = lm(Tensor(X))
ggn = ggn_matrix(out, params, lambda u: cross_entropy_hvp(out.data, u))
precision = laplace_posterior_precision(ggn, n_data, prior_prec)
chol = np.linalg.cholesky(precision)
thetas = sample(theta_map, chol, 2000, seed=0)
print("eigenvalues of Lambda", np.round(np.linalg.eigvalsh(precision), 2))


def predictive(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MAP and linearized Laplace probability of class 1 at the points, plus the Laplace std."""
    out = lm(Tensor(points))
    jac = jacobian(out, params)  # M, K, P
    logits = out.data[..., None] + jac @ (thetas - theta_map).T  # M, K, S
    p1 = 1 / (1 + np.exp(-np.clip(logits[:, 1] - logits[:, 0], -50, 50)))  # M, S
    p1_map = 1 / (1 + np.exp(-np.clip(out.data[:, 1] - out.data[:, 0], -50, 50)))
    return p1_map, p1.mean(axis=1), p1.std(axis=1)


# %%
# MAP versus Laplace on the training data
p_map, p_lap, _ = predictive(X)
for name, p1 in [("MAP", p_map), ("Laplace", p_lap)]:
    p1 = np.clip(p1, 1e-12, 1 - 1e-12)
    nll = -np.mean(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    acc = np.mean((p1 > 0.5) == y)
    print(f"{name}: accuracy {acc:.3f}, NLL {nll:.4f}, Brier {np.mean((p1 - y) ** 2):.4f}")

# %%
xx, yy, points = grid(X)
p_map, p_lap, p_std = predictive(points)

fig, axes = plt.subplots(1, 3, sharey=True)
panels = [(p_map, "MAP", "RdBu_r"), (p_lap, "Laplace", "RdBu_r"), (p_std, "std of p, Monte Carlo", "viridis")]
for ax, (z, title, cmap) in zip(axes, panels, strict=True):
    cs = ax.contourf(xx, yy, z.reshape(xx.shape), levels=11, cmap=cmap)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", s=3, edgecolors="none")
    ax.set_title(title)
    fig.colorbar(cs, ax=ax)
fig.savefig(FIGURES / "linear_laplace.png", bbox_inches="tight")
plt.show()


# %%
# Evidence against the prior precision, with a fresh Newton MAP for every tau
def fit_map(prior_prec: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Newton from zeros for one tau. Returns the mean loss, theta_map and the Cholesky factor."""
    for p in params:
        p.data[...] = 0.0
    for _ in range(30):
        out = lm(Tensor(X))
        g = autograd.vjp(cross_entropy_loss(out, Y))
        theta = flatten([p.data for p in params])
        grad_mean = flatten([g[p] for p in params])
        if np.linalg.norm(n_data * grad_mean + prior_prec * theta) < 1e-10:
            break
        ggn = ggn_matrix(out, params, lambda u: cross_entropy_hvp(out.data, u))
        theta = newton_step(theta, grad_mean, ggn, n_data, prior_prec)
        for p, piece in zip(params, unflatten(theta, like), strict=True):
            p.data[...] = piece
    out = lm(Tensor(X))
    ggn = ggn_matrix(out, params, lambda u: cross_entropy_hvp(out.data, u))
    chol = np.linalg.cholesky(laplace_posterior_precision(ggn, n_data, prior_prec))
    loss = float(cross_entropy_loss(out, Y).data)
    return loss, flatten([p.data for p in params]), chol


taus = np.logspace(-3, 2, 26)
values = []
for tau in taus:
    loss_tau, theta_tau, chol_tau = fit_map(tau)
    values.append(log_marginal_likelihood(loss_tau, theta_tau, chol_tau, n_data, tau))
log_z = np.array(values)
best = taus[np.argmax(log_z)]
print(f"evidence is largest at tau = {best:.3g}")

fig, ax = plt.subplots()
ax.semilogx(taus, log_z, marker="o", markersize=3)
ax.axvline(best, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel(r"prior precision $\tau$")
ax.set_ylabel(r"$\log p(D \mid \tau)$, Laplace")
ax.set_title("Evidence of the linear model")
fig.savefig(FIGURES / "linear_evidence.png", bbox_inches="tight")
plt.show()
