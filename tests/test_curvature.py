import numpy as np
import pytest
from numpy.testing import assert_allclose

from curvgrad import autograd
from curvgrad.curvature.ggn import flatten, ggn_matrix, ggn_vp, unflatten
from curvgrad.nn.layers import Linear
from curvgrad.nn.loss import cross_entropy_hvp, cross_entropy_loss
from curvgrad.tensor import Tensor

CASES = [(2, 2), (3, 3)]  # (input dim d, classes K)


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _setup(d: int, K: int, N: int = 40, seed: int = 0):
    """Random data and a linear softmax model with generic weights. Returns X, Y, model."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, d))
    Y = np.eye(K)[rng.integers(0, K, size=N)]
    model = Linear(d, K)
    model.W.data = rng.normal(size=(d, K))
    model.b.data = rng.normal(size=K)
    return X, Y, model


def _ggn_args(model, X):
    """Build the graph once and return the arguments of ggn_vp except v."""
    out = model(Tensor(X))
    return out, model.parameters(), lambda u: cross_entropy_hvp(out.data, u)


# The oracles below are written without flatten, unflatten, jvp or the loss Hessian, so they stay
# independent of the code under test.


def _theta(model) -> np.ndarray:
    return np.concatenate([p.data.ravel() for p in model.parameters()])


def _set_theta(model, theta: np.ndarray) -> None:
    i = 0
    for p in model.parameters():
        p.data = theta[i : i + p.data.size].reshape(p.data.shape).copy()
        i += p.data.size


def _grad(model, X, Y) -> np.ndarray:
    g = autograd.vjp(cross_entropy_loss(model(Tensor(X)), Y))
    return np.concatenate([g[p].ravel() for p in model.parameters()])


def fd_hessian(model, X, Y, h: float = 1e-6) -> np.ndarray:
    """Exact Hessian of the loss in the parameters by central differences of the gradient.

    Column i is (grad(theta + h e_i) - grad(theta - h e_i)) / 2h. The truncation error is O(h^2)
    and the roundoff error is O(eps / h), so h near 1e-6 is close to the optimum in float64.
    """
    theta = _theta(model)
    H = np.empty((theta.size, theta.size))
    for i in range(theta.size):
        e = np.zeros_like(theta)
        e[i] = h
        _set_theta(model, theta + e)
        g_plus = _grad(model, X, Y)
        _set_theta(model, theta - e)
        g_minus = _grad(model, X, Y)
        H[:, i] = (g_plus - g_minus) / (2 * h)
    _set_theta(model, theta)
    return 0.5 * (H + H.T)


def dense_ggn(model, X) -> np.ndarray:
    """G = sum_n J_n^T H_n J_n for the linear softmax model, assembled with plain numpy.

    With W flattened in C order the logits of sample n have the Jacobian
    J_n = [kron(x_n^T, I_K), I_K] of shape K by (d K + K). H_n = (diag(p) - p p^T) / N.
    """
    d, K = model.W.shape
    N = X.shape[0]
    probs = _softmax(X @ model.W.data + model.b.data)
    I = np.eye(K)
    G = np.zeros((d * K + K, d * K + K))
    for n in range(N):
        J = np.hstack([np.kron(X[n][None, :], I), I])
        H_n = (np.diag(probs[n]) - np.outer(probs[n], probs[n])) / N
        G += J.T @ H_n @ J
    return G


def test_flatten_roundtrip():
    rng = np.random.default_rng(0)
    arrays = [rng.normal(size=(2, 3)), rng.normal(size=(3,)), rng.normal(size=())]
    vec = flatten(arrays)
    assert vec.shape == (10,)
    assert_allclose(vec[:6], arrays[0].ravel(), rtol=0, atol=0)
    for a, b in zip(arrays, unflatten(vec, arrays), strict=True):
        assert a.shape == b.shape
        assert_allclose(a, b, rtol=0, atol=0)


def test_cross_entropy_hvp_is_the_hessian_of_the_loss():
    # The gradient of the mean cross entropy with respect to the logits is (p - Y) / N.
    rng = np.random.default_rng(1)
    Z, U = rng.normal(size=(6, 4)), rng.normal(size=(6, 4))
    Y = np.eye(4)[rng.integers(0, 4, size=6)]
    h = 1e-6
    fd = ((_softmax(Z + h * U) - Y) - (_softmax(Z - h * U) - Y)) / (2 * h * 6)
    assert_allclose(cross_entropy_hvp(Z, U), fd, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("d,K", CASES)
def test_ggn_vp_matches_dense_oracle(d, K):
    X, _, model = _setup(d, K)
    G = dense_ggn(model, X)
    v = np.random.default_rng(1).normal(size=G.shape[0])
    assert_allclose(ggn_vp(*_ggn_args(model, X), v), G @ v, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("d,K", CASES)
def test_ggn_equals_hessian_for_a_linear_model(d, K):
    # The logits are linear in the parameters, so the second order term of the Hessian vanishes.
    X, Y, model = _setup(d, K)
    G = ggn_matrix(*_ggn_args(model, X))
    assert G.shape == (d * K + K,) * 2
    assert_allclose(G, fd_hessian(model, X, Y), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("d,K", CASES)
def test_ggn_is_symmetric_and_positive_semidefinite(d, K):
    X, _, model = _setup(d, K)
    G = ggn_matrix(*_ggn_args(model, X))
    assert_allclose(G, G.T, atol=1e-12)
    assert np.linalg.eigvalsh(G).min() > -1e-12


@pytest.mark.parametrize("d,K", CASES)
def test_ggn_rank_and_null_space(d, K):
    # Softmax ignores adding the same affine function u.x + t to every logit, which gives a
    # null space of dimension d + 1 inside the (d + 1) K parameters.
    X, _, model = _setup(d, K)
    G = ggn_matrix(*_ggn_args(model, X))
    eig = np.linalg.eigvalsh(G)
    assert int((eig > 1e-10 * eig.max()).sum()) == (d + 1) * (K - 1)

    rng = np.random.default_rng(2)
    u, t = rng.normal(size=d), rng.normal()
    null = flatten([np.tile(u[:, None], (1, K)), np.full(K, t)])
    assert_allclose(G @ null, 0.0, atol=1e-10)


def test_ggn_vp_is_linear_and_reuses_the_graph():
    X, _, model = _setup(2, 2)
    args = _ggn_args(model, X)
    rng = np.random.default_rng(3)
    v1, v2 = rng.normal(size=6), rng.normal(size=6)
    combined = ggn_vp(*args, 2.0 * v1 - 3.0 * v2)
    separate = 2.0 * ggn_vp(*args, v1) - 3.0 * ggn_vp(*args, v2)
    assert_allclose(combined, separate, rtol=1e-12, atol=1e-14)
