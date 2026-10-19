import numpy as np
import pytest

from curvgrad.autograd import _unbroadcast
from curvgrad.tensor import Tensor

# g lives in the broadcast (big) shape t, x lives in the target (small) shape s.
# (g_shape, target_shape)
SHAPE_PAIRS = [
    ((3, 4), (3, 4)),
    ((3, 4), (4,)),
    ((3, 4), (1, 4)),
    ((3, 4), ()),
    ((2, 3, 4), (3, 1)),
    ((2, 3, 4), (4,)),
]


def _grad(t: Tensor) -> np.ndarray:
    # .grad is None until a gradient arrives
    assert t.grad is not None
    return t.grad


@pytest.mark.parametrize(("g_shape", "target"), SHAPE_PAIRS)
def test_unbroadcast_is_adjoint_of_broadcast(g_shape, target):
    # Broadcasting is a linear map B, so un-broadcasting must be its adjoint B^T, and the adjoint
    # is the unique map satisfying <g, Bx> = <B^T g, x> for all g, x. 
    rng = np.random.default_rng(0) 
    g = rng.standard_normal(g_shape)
    x = rng.standard_normal(target)

    lhs = np.sum(g * np.broadcast_to(x, g_shape))  # <g, Bx>
    rhs = np.sum(_unbroadcast(g, target) * x)  # <B^T g, x>

    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


@pytest.mark.parametrize(("g_shape", "target"), SHAPE_PAIRS)
def test_unbroadcast_returns_exact_target_shape(g_shape, target):
    # The adjoint identity alone does not pin the shape: returning (1,4) where (4,) is wanted gives
    # an identical sum, because the product broadcasts back. So shape needs its own assertion.
    assert _unbroadcast(np.ones(g_shape), target).shape == target


def test_unbroadcast_sums_the_fanned_out_axis():
    # grounds the derivation in autograd.py: W_j feeds 2 rows, so each gradient is a sum of 2 ones
    np.testing.assert_allclose(_unbroadcast(np.ones((2, 3)), (3,)), [2.0, 2.0, 2.0])


def test_fan_out_accumulates():
    # z = x*y + x: x has two consumers, so its cotangent is the sum of two contributions.
    # Wrong accumulation (= instead of +=) gives 3.0; wrong topological order gives 1.0.
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    z = x * y + x
    z.backward()
    assert z.data == 8.0
    assert x.grad == 4.0
    assert y.grad == 2.0


def test_self_edge_is_not_deduplicated():
    # _inputs must be an ordered tuple: a set would collapse (x, x) to one element and drop an edge
    x = Tensor(3.0, requires_grad=True)
    (x + x).backward()
    assert x.grad == 2.0

    w = Tensor(3.0, requires_grad=True)
    (w * w).backward()
    assert w.grad == 6.0


def test_broadcast_bias_gradient_has_input_shape():
    # the case that blocks every Linear layer: bias (4,) added to activations (3,4)
    a = Tensor(np.ones((3, 4)), requires_grad=True)
    b = Tensor(np.ones(4), requires_grad=True)
    (a + b).backward(np.ones((3, 4)))
    assert _grad(b).shape == (4,)
    np.testing.assert_allclose(_grad(b), 3.0)
    assert _grad(a).shape == (3, 4)


def test_backward_twice_accumulates():
    # correct semantics and what makes zero_grad mean something
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    (x * y).backward()
    (x * y).backward()
    assert x.grad == 6.0
    x.zero_grad()
    assert x.grad is None


def test_non_scalar_backward_needs_a_seed():
    x = Tensor(np.ones(3), requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2.0).backward()


def test_reduction_over_a_non_trailing_axis():
    # sum(axis=1) leaves a (3,) cotangent, which cannot broadcast against (3,4) until the reduced
    # axis is put back 
    a = Tensor(np.arange(12, dtype=float).reshape(3, 4), requires_grad=True)
    a.sum(axis=1).backward(np.ones(3))
    np.testing.assert_allclose(_grad(a), np.ones((3, 4)))

    m = Tensor(np.arange(12, dtype=float).reshape(3, 4), requires_grad=True)
    m.mean(axis=1).backward(np.ones(3))
    np.testing.assert_allclose(_grad(m), np.full((3, 4), 0.25))


def test_scalar_operands_are_wrapped():
    x = Tensor(2.0, requires_grad=True)
    (x * 3.0).backward()
    assert x.grad == 3.0
    y = Tensor(2.0, requires_grad=True)
    (3.0 - y).backward()
    assert y.grad == -1.0


def test_numpy_does_not_hijack_left_operand():
    # without __array_ufunc__ = None this silently returns an object-dtype ndarray
    out = np.ones(3) + Tensor(np.ones(3), requires_grad=True)
    assert isinstance(out, Tensor)


def test_softmax_vjp_matches_explicit_jacobian():
    # J = diag(p) - p p^T. the rule must agree with building J and transposing it
    rng = np.random.default_rng(1)
    logits = rng.standard_normal(5)
    g = rng.standard_normal(5)

    a = Tensor(logits, requires_grad=True)
    a.softmax().backward(g)

    p = np.exp(logits - logits.max())
    p /= p.sum()
    expected = (np.diag(p) - np.outer(p, p)).T @ g
    np.testing.assert_allclose(_grad(a), expected, rtol=1e-12)
