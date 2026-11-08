# %%
import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles

from curvgrad import autograd
from curvgrad.curvature.ggn import ggn_matrix
from curvgrad.nn.layers import Linear
from curvgrad.nn.loss import cross_entropy_hvp, cross_entropy_loss
from curvgrad.tensor import Tensor

plt.rcParams.update(bundles.tue_ai_beamer())
plt.rcParams.update({"figure.dpi": 150})


def generate_two_moons(n_samples=100, noise=0.1, random_state=None):

    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    theta_out = rng.uniform(0, np.pi, n_samples_out)
    x_out = np.cos(theta_out)
    y_out = np.sin(theta_out)

    theta_in = rng.uniform(0, np.pi, n_samples_in)
    x_in = 1.0 - np.cos(theta_in)
    y_in = 0.5 - np.sin(theta_in)

    X = np.vstack(
        [np.column_stack((x_out, y_out)), np.column_stack((x_in, y_in))]
    )
    y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])

    if noise > 0:
        X += rng.normal(scale=noise, size=X.shape)

    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


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
G = ggn_matrix(out, lm.parameters(), lambda u: cross_entropy_hvp(out.data, u))
labels = ["W00", "W01", "W10", "W11", "b0", "b1"]

fig, (ax1, ax2) = plt.subplots(1, 2)
im = ax1.imshow(G, cmap="RdBu_r", vmin=-np.abs(G).max(), vmax=np.abs(G).max())
ax1.set_xticks(range(6), labels)
ax1.set_yticks(range(6), labels)
ax1.set_title("GGN")
fig.colorbar(im, ax=ax1)
ax2.bar(range(6), np.clip(np.linalg.eigvalsh(G), 1e-16, None))
ax2.set_yscale("log")
ax2.set_title("eigenvalues")
plt.show()
