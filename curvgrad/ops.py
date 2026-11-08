# Atomic mathematical operations. Imports Tensor lazily inside _make to avoid the circular import
# with tensor.py (tensor.py needs ops for its operator overloads, ops needs Tensor to build nodes).
# Each op returns a Tensor holding the forward value plus the LOCAL backward rule of that op.
# Picking up the graphical view from tensor.py: an op creates a node that records the edge back to
# its inputs, and for each the derivative is clearly specified. The actual autograd program is
# outsourced to autograd.py, since broadcasting and traversing the graph require more logic than
# any single op knows about.
#
# A rule is `vjp(g) -> tuple`, one cotangent per input, positionally aligned with `inputs`, and
# BEFORE un-broadcasting: autograd.vjp does the reduction and the accumulation, so no op here ever
# accumulates anything. That keeps every shape/accumulation concern in exactly one place.
#
# `from __future__ import annotations` makes annotations plain strings so they are not evaluated at
# import time -- Tensor only has to exist when the function is CALLED, which is why the local import
# is enough. The TYPE_CHECKING block additionally lets mypy resolve the name with no runtime import.

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from curvgrad.tensor import Tensor

VJPRule = Callable[[np.ndarray], tuple[np.ndarray | None, ...]]


def _make(
    data: ArrayLike, 
    inputs: tuple[Tensor, ...],
    vjp: VJPRule,
    op: str,
    jvp: Callable[..., np.ndarray] | None = None,
) -> Tensor:
    """Wrap a forward value plus its local rules into a graph node."""
    from curvgrad.tensor import Tensor

    out = Tensor(data, requires_grad=any(t.requires_grad for t in inputs))
    out._inputs = tuple(inputs)
    out._vjp = vjp
    out._jvp = jvp  
    out._op = op
    return out


def _restore_axes(g: np.ndarray, shape: tuple[int, ...], axis) -> np.ndarray:
    """Adjoint of a reduction: re-insert the axes it removed, then broadcast back to `shape`.

    numpy aligns from the right, so a (3,) cotangent from sum(axis=1) on a (3,4) input cannot
    broadcast against (3,4) until the reduced axis is put back.
    """
    if axis is None:
        return np.broadcast_to(g, shape)
    return np.broadcast_to(np.expand_dims(g, axis), shape)


def add(a: Tensor, b: Tensor) -> Tensor:
    def vjp(g: np.ndarray):
        return (g, g)

    def jvp(da: np.ndarray, db: np.ndarray) -> np.ndarray:
        return da + db

    return _make(a.data + b.data, (a, b), vjp, "add", jvp)


def sub(a: Tensor, b: Tensor) -> Tensor:
    def vjp(g: np.ndarray):
        return (g, -g)

    def jvp(da: np.ndarray, db: np.ndarray) -> np.ndarray:
        return da - db

    return _make(a.data - b.data, (a, b), vjp, "sub", jvp)


def mul(a: Tensor, b: Tensor) -> Tensor:
    def vjp(g: np.ndarray):
        return (b.data * g, a.data * g)

    def jvp(da: np.ndarray, db: np.ndarray) -> np.ndarray:
        return da * b.data + a.data * db

    return _make(a.data * b.data, (a, b), vjp, "mul", jvp)


def div(a: Tensor, b: Tensor) -> Tensor:
    def vjp(g: np.ndarray):
        return (g / b.data, -g * a.data / b.data**2)

    def jvp(da: np.ndarray, db: np.ndarray) -> np.ndarray:
        return da / b.data - a.data * db / b.data**2

    return _make(a.data / b.data, (a, b), vjp, "div", jvp)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    # 2-D only for now
    def vjp(g: np.ndarray):
        return (g @ b.data.T, a.data.T @ g)

    def jvp(da: np.ndarray, db: np.ndarray) -> np.ndarray:
        return da @ b.data + a.data @ db

    return _make(a.data @ b.data, (a, b), vjp, "matmul", jvp)


def sum_(a: Tensor, axis=None) -> Tensor:
    def vjp(g: np.ndarray):
        return (_restore_axes(g, a.data.shape, axis),)

    def jvp(da: np.ndarray) -> np.ndarray:
        return np.asarray(np.sum(da, axis=axis))

    return _make(np.sum(a.data, axis=axis), (a,), vjp, "sum", jvp)


def mean_(a: Tensor, axis=None) -> Tensor:
    out_data = np.mean(a.data, axis=axis)
    n = a.data.size // np.size(out_data)

    def vjp(g: np.ndarray):
        return (_restore_axes(g, a.data.shape, axis) / n,)

    def jvp(da: np.ndarray) -> np.ndarray:
        return np.asarray(np.mean(da, axis=axis))

    return _make(out_data, (a,), vjp, "mean", jvp)


def reshape(a: Tensor, shape: tuple[int, ...]) -> Tensor:
    def vjp(g: np.ndarray):
        return (g.reshape(a.data.shape),)

    def jvp(da: np.ndarray) -> np.ndarray:
            return da.reshape(shape)

    return _make(a.data.reshape(shape), (a,), vjp, "reshape", jvp)


def neg(a: Tensor) -> Tensor:
    def vjp(g: np.ndarray):
        return (-g,)

    def jvp(da: np.ndarray) -> np.ndarray:
        return -da

    return _make(-a.data, (a,), vjp, "neg", jvp)


def relu(a: Tensor) -> Tensor:
    def vjp(g: np.ndarray):
        return ((a.data > 0) * g,)

    def jvp(da: np.ndarray) -> np.ndarray:
            return (a.data > 0) * da

    return _make(np.maximum(0, a.data), (a,), vjp, "relu", jvp)


def sigmoid(a: Tensor) -> Tensor:
    s = 1 / (1 + np.exp(-a.data))

    def vjp(g: np.ndarray):
        return (s * (1 - s) * g,)

    def jvp(da: np.ndarray) -> np.ndarray:
            return s * (1 - s) * da

    return _make(s, (a,), vjp, "sigmoid", jvp)


def softmax(a: Tensor) -> Tensor:
    shift = a.data - np.max(a.data, axis=-1, keepdims=True) 
    exp = np.exp(shift)
    p = exp / np.sum(exp, axis=-1, keepdims=True)

    def vjp(g: np.ndarray):
        # J = diag(p) - p p^T, so J^T g = p * (g - p.g)
        return (p * (g - np.sum(p * g, axis=-1, keepdims=True)),)

    def jvp(da: np.ndarray) -> np.ndarray:
        return p * (da - np.sum(p * da, axis=-1, keepdims=True))

    return _make(p, (a,), vjp, "softmax", jvp)


def exp(a: Tensor) -> Tensor:
    e = np.exp(a.data)

    def vjp(g: np.ndarray):
        return (g * e, )

    def jvp(da: np.ndarray) -> np.ndarray:
        return da * e
    
    return _make(e, (a,), vjp, "exp", jvp)


def log(a: Tensor) -> Tensor:
    def vjp(g: np.ndarray):
        return (g / a.data, )

    def jvp(da: np.ndarray) -> np.ndarray:
        return da / a.data

    return _make(np.log(a.data), (a,), vjp, "log", jvp)


def log_softmax(a: Tensor) -> Tensor:
    shift = a.data - np.max(a.data, axis=-1, keepdims=True)
    p = shift - np.log(np.sum(np.exp(shift), axis=-1, keepdims=True))
    s = np.exp(p)

    def vjp(g: np.ndarray):
        return (g - s * np.sum(g, axis=-1, keepdims=True), )

    def jvp(da: np.ndarray) -> np.ndarray:
        return da - np.sum(s * da, axis=-1, keepdims=True)

    return _make(p, (a,), vjp, "log_softmax", jvp)


def transpose(a: Tensor) -> Tensor:
    def vjp(g: np.ndarray):
        return (g.T,)

    def jvp(da: np.ndarray) -> np.ndarray:
            return da.T

    return _make(a.data.T, (a,), vjp, "transpose", jvp)

def pow_(a: Tensor, b: float) -> Tensor: #TODO: Edge case with b=0 at a=0
    def vjp(g: np.ndarray):
        return (g * b * a.data ** (b - 1),)

    def jvp(da: np.ndarray) -> np.ndarray:
        return  da * b * a.data ** (b - 1)

    return _make(a.data ** b, (a,), vjp, "pow", jvp)

def abs_(a: Tensor) -> Tensor:
    def vjp(g: np.ndarray):
        return (g * np.sign(a.data),)

    def jvp(da: np.ndarray) -> np.ndarray:
        return da * np.sign(a.data)

    return _make(np.abs(a.data), (a,), vjp, "abs", jvp)
