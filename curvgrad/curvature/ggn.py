from collections.abc import Callable, Sequence

import numpy as np

from curvgrad.tensor import Tensor


def flatten(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate the arrays, each raveled in C order, into one 1D vector."""
    raise NotImplementedError


def unflatten(vec: np.ndarray, like: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Split vec into arrays with the shapes of like. Inverse of flatten."""
    raise NotImplementedError


def ggn_vp(
    output: Tensor,
    params: Sequence[Tensor],
    loss_hvp: Callable[[np.ndarray], np.ndarray],
    v: np.ndarray,
) -> np.ndarray:
    """Return G v for the generalized Gauss Newton matrix G = J^T H J.

    The output is the model output node, here the logits, built from the current parameter values.
    J is the Jacobian of the output with respect to the flat parameter vector. H is the Hessian of
    the loss with respect to the output and is only available as the product loss_hvp(u).

    Three steps:
    1. u = J v. Unflatten v, seed autograd.jvp on the leaves in params, read the tangent at output.
    2. w = H u. Call loss_hvp(u).
    3. G v = J^T w. Call autograd.vjp on output with w as the seed, then flatten the cotangents of
       params in the same order. A parameter without a cotangent gets zeros.

    The graph is only read, so it can be reused for many v. Do not change p.data between calls.
    """
    raise NotImplementedError


def ggn_matrix(
    output: Tensor,
    params: Sequence[Tensor],
    loss_hvp: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Return the dense G by applying ggn_vp to each standard basis vector. Column i is G e_i.

    Only for small models, tests and plots.
    """
    raise NotImplementedError
