import numpy as np
import pytest

from curvgrad import autograd, ops
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


def fd_directional(f, xs, vs, h=1e-6):
    """Central difference of f in direction vs, perturbing all inputs together."""
    plus = f(*[x + h * v for x, v in zip(xs, vs)])
    minus = f(*[x - h * v for x, v in zip(xs, vs)])
    return (plus - minus) / (2 * h)


def _raw(op):
    """op on Tensors -> function on raw arrays, so the oracle shares no autodiff code with op."""

    def f(*arrs):
        return op(*[Tensor(a) for a in arrs]).data

    return f


def check_vjp(op, xs, rng, n_dirs=3):
    # <u, J v> (finite difference) must equal sum_i <vjp(u)_i, v_i>, since <u, Jv> = <J^T u, v>.
    # Goes through autograd.vjp, NOT the raw node._vjp, so _unbroadcast is tested as well.
    leaves = [Tensor(x, requires_grad=True) for x in xs]
    out = op(*leaves)
    for _ in range(n_dirs):  # several random directions: one could be unlucky
        vs = [rng.standard_normal(x.shape) for x in xs]
        u = rng.standard_normal(out.shape)
        fd = fd_directional(_raw(op), xs, vs)
        cots = autograd.vjp(out, u)
        rhs = sum(np.sum(cots[leaf] * v) for leaf, v in zip(leaves, vs))
        np.testing.assert_allclose(rhs, np.sum(u * fd), rtol=1e-6, atol=1e-8)


def check_jvp(op, xs, rng, n_dirs=3):
    # the forward rule against the finite difference, elementwise on the full output array
    out = op(*[Tensor(x) for x in xs])
    assert out._jvp is not None, f"op {out._op!r} has no _jvp rule yet"
    for _ in range(n_dirs):
        vs = [rng.standard_normal(x.shape) for x in xs]
        fd = fd_directional(_raw(op), xs, vs)
        np.testing.assert_allclose(out._jvp(*vs), fd, rtol=1e-6, atol=1e-8)


def check_op(op, xs, rng, n_dirs=3):
    check_vjp(op, xs, rng, n_dirs)
    check_jvp(op, xs, rng, n_dirs)


# Input samplers
def _normal(*shapes):
    return lambda rng: [rng.standard_normal(s) for s in shapes]


def _positive(*shapes):  # log, fractional pow, denominators
    return lambda rng: [rng.uniform(0.5, 2.0, s) for s in shapes]


def _off_zero(*shapes):  # relu / abs: finite differences across the kink at 0 are meaningless
    def sample(rng):
        xs = [rng.standard_normal(s) for s in shapes]
        return [x + 0.3 * np.sign(x) for x in xs]  # |x| >= 0.3 >> h*|v|

    return sample


def _div_inputs(rng):
    return [rng.standard_normal((3, 4)), rng.uniform(0.5, 2.0, (4,))]


# (op on Tensors, sampler). Binary ops use a broadcast pair so _unbroadcast sits under the check.
VJP_CASES = [
    pytest.param(ops.add, _normal((3, 4), (4,)), id="add-bcast-row"),
    pytest.param(ops.add, _normal((3, 4), (3, 1)), id="add-bcast-col"),
    pytest.param(ops.sub, _normal((3, 4), (4,)), id="sub-bcast"),
    pytest.param(ops.mul, _normal((3, 4), (4,)), id="mul-bcast"),
    pytest.param(ops.div, _div_inputs, id="div-bcast"),
    pytest.param(ops.matmul, _normal((2, 3), (3, 4)), id="matmul"),
    pytest.param(lambda a: ops.sum_(a), _normal((3, 4)), id="sum-all"),
    pytest.param(lambda a: ops.sum_(a, axis=1), _normal((3, 4)), id="sum-axis1"),
    pytest.param(lambda a: ops.mean_(a, axis=0), _normal((3, 4)), id="mean-axis0"),
    pytest.param(lambda a: ops.reshape(a, (6,)), _normal((2, 3)), id="reshape"),
    pytest.param(ops.transpose, _normal((3, 4)), id="transpose"),
    pytest.param(ops.neg, _normal((5,)), id="neg"),
    pytest.param(ops.relu, _off_zero((3, 4)), id="relu"),
    pytest.param(ops.abs_, _off_zero((3, 4)), id="abs"),
    pytest.param(ops.sigmoid, _normal((3, 4)), id="sigmoid"),
    pytest.param(ops.exp, _normal((3, 4)), id="exp"),
    pytest.param(ops.log, _positive((3, 4)), id="log"),
    pytest.param(lambda a: ops.pow_(a, 3), _normal((3, 4)), id="pow-int"),
    pytest.param(lambda a: ops.pow_(a, 0.5), _positive((3, 4)), id="pow-frac"),
    pytest.param(ops.softmax, _normal((3, 5)), id="softmax"),
    pytest.param(ops.log_softmax, _normal((3, 5)), id="log_softmax"),
]


@pytest.mark.parametrize(("op", "sample"), VJP_CASES)
def test_vjp_matches_finite_differences(op, sample):
    rng = np.random.default_rng(0)
    check_vjp(op, sample(rng), rng)


# Forward-mode rules
JVP_CASES = [
    pytest.param(ops.add, _normal((3, 4), (4,)), id="add-bcast-row"),
    pytest.param(ops.add, _normal((3, 4), (3, 1)), id="add-bcast-col"),
    pytest.param(ops.add, _normal((3, 4), ()), id="add-scalar"),
]


@pytest.mark.parametrize(("op", "sample"), JVP_CASES)
def test_jvp_matches_finite_differences(op, sample):
    rng = np.random.default_rng(0)
    check_jvp(op, sample(rng), rng)


# tests of the helper itself

def test_fd_directional_on_a_known_function():
    # f(x) = x^2  ->  directional derivative 2 x v
    rng = np.random.default_rng(0)
    x, v = rng.standard_normal(5), rng.standard_normal(5)
    np.testing.assert_allclose(fd_directional(lambda a: a**2, [x], [v]), 2 * x * v, rtol=1e-6)


def test_helper_catches_a_wrong_vjp():
    # exp bug: the cotangent was exponentiated instead of multiplied
    def bad_exp(a):
        e = np.exp(a.data)
        return ops._make(e, (a,), lambda g: (np.exp(g),), "exp", lambda da: da * e)

    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError):
        check_vjp(bad_exp, [rng.standard_normal(5)], rng)


def test_helper_catches_a_wrong_jvp():
    def bad_add(a, b):  # correct vjp, wrong sign in the jvp
        out = ops.add(a, b)
        out._jvp = lambda da, db: da - db
        return out

    rng = np.random.default_rng(0)
    xs = [rng.standard_normal((3, 4)), rng.standard_normal(4)]
    with pytest.raises(AssertionError):
        check_jvp(bad_add, xs, rng)


def test_helper_catches_a_missing_unbroadcast():
    # the VJP check goes through the engine, so it also covers shape handling
    def bad_add(a, b):
        out = ops.add(a, b)
        out._vjp = lambda g: (g, g * 2.0)  # wrong scale on b
        return out

    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError):
        check_vjp(bad_add, [rng.standard_normal((3, 4)), rng.standard_normal(4)], rng)

