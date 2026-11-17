from collections.abc import Callable, Sequence

import numpy as np

from curvgrad import autograd
from curvgrad.tensor import Tensor


def flatten(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate the arrays into one flat vector, each raveled in C order."""
    return np.concatenate([np.ravel(array) for array in arrays])


def unflatten(vec: np.ndarray, like: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Split a flat vector into arrays shaped like `like`. Inverse of flatten."""
    ends = np.cumsum([array.size for array in like])
    pieces = np.split(vec, ends[:-1])
    return [piece.reshape(array.shape) for piece, array in zip(pieces, like, strict=True)]


def ggn_vp(
    output: Tensor,
    params: Sequence[Tensor],
    loss_hvp: Callable[[np.ndarray], np.ndarray],
    v: np.ndarray,
) -> np.ndarray:
    """Matrix-free GGN product G v = J^T H J v.

    J is the Jacobian of `output` with respect to the flat parameters. H is the loss Hessian with
    respect to `output`, available only through `loss_hvp`.
    """
    v_unf = unflatten(v, [p.data for p in params])
    Jv = autograd.jvp(output, dict(zip(params, v_unf, strict=True)))[output]  # J v
    w = loss_hvp(Jv)  # H J v
    cot = autograd.vjp(output, w)  # J^T H J v, per parameter

    return flatten([cot.get(p, np.zeros_like(p.data)) for p in params])


def ggn_matrix(
    output: Tensor,
    params: Sequence[Tensor],
    loss_hvp: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Dense G, one ggn_vp per basis vector. Only for small models."""
    P = sum(p.data.size for p in params)
    lookup = np.eye(P)
    columns = []
    for i in range(P):
        e = lookup[:, i]
        columns.append(ggn_vp(output, params, loss_hvp, e))
    return np.stack(columns, axis=1)


def jacobian(
    output: Tensor,
    params: Sequence[Tensor],
) -> np.ndarray:
    """Jacobian of `output` with respect to the flat parameters, one jvp per column. Shape M by K by P."""
    P = sum(p.data.size for p in params)
    lookup = np.identity(P)
    columns = []
    for i in range(P):
        e = unflatten(lookup[:, i], [p.data for p in params])
        columns.append(autograd.jvp(output, dict(zip(params, e, strict=True)))[output])

    return np.stack(columns, axis=-1)
