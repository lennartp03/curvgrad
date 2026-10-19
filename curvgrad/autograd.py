# Broadcasting: numpy aligns shapes from the right and every axis pair must either match or be 1.
# e.g. x.shape = (2,3) and W.shape = (3,) -> W is expanded along the axis it does NOT have (i),
# not along the matching one (j):
#   y_ij = x_ij + W_j,    i in (0,1), j in (0,1,2)
# Now ask for dL/dW_j = sum_i dL/dy_ij * dy_ij/dW_j -- W_j feeds 2 different outputs, so the chain
# rule sums over both. That sum IS the un-broadcast: broadcasting is implicit fan-out, and fan-out
# means summation in reverse mode (same reason x accumulates twice in z = x*y + x).
# Mechanically numpy left-pads 1s until the ranks match, then repeats along every axis where the
# padded shape has 1 and the target has n. The adjoint therefore sums over exactly those axes:
# axes that padding invented are dropped, axes that already existed as 1 are kept as 1.
#
# This file also holds the backprop algorithm itself: topological sort + reverse traversal +
# accumulation. ops.py holds only the LOCAL rules and tensor.py only the node, because
# backprop = local derivative rules composed along the graph. Nielsen's BP1-BP4 fuses those two
# because his graph is a fixed chain; splitting them is what generalises to an arbitrary DAG.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from curvgrad.tensor import Tensor


def _unbroadcast(g: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Adjoint of broadcasting: reduce `g` to exactly `shape` by summing the fanned-out axes."""

    assert g.ndim >= len(shape), f"cotangent rank {g.ndim} below target rank {len(shape)}"

    if g.shape == shape:
        return g

    # phase 1: axes numpy prepended
    n_pad = g.ndim - len(shape)
    if n_pad:
        g = g.sum(axis=tuple(range(n_pad)))

    # phase 2: axes that existed as 1 and were repeated
    singleton = tuple(k for k, n in enumerate(shape) if n == 1 and g.shape[k] != 1)
    if singleton:
        g = g.sum(axis=singleton, keepdims=True)

    return np.asarray(g)


def _topological_order(root: Tensor) -> list[Tensor]:
    """Every node reachable from `root`, ordered so each node appears AFTER all of its inputs."""
    order: list[Tensor] = []
    visited: set[int] = set()  # keyed on id(), so it does not depend on Tensor.__hash__

    def visit(node: Tensor) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        for inp in node._inputs:
            visit(inp)
        order.append(node)  # inputs-first order

    visit(root)  # recursive: depth is O(graph depth)
    return order


def vjp(root: Tensor, seed: np.ndarray | None = None) -> dict[Tensor, np.ndarray]:
    """Reverse-mode sweep. Returns {node: cotangent} without touching any .grad.

    Stateless by design: two calls with different seeds do not interfere, which is what
    matrix-free curvature-vector products need later (they push many v through one graph).
    """
    if seed is None:
        if root.shape != ():
            raise RuntimeError("backward() on a non-scalar output needs an explicit gradient")
        seed = np.ones_like(root.data)
    seed = np.asarray(seed, dtype=np.float64)
    if seed.shape != root.shape:
        raise ValueError(f"seed shape {seed.shape} does not match output shape {root.shape}")

    # Tensor is hashable by identity only because it defines no __eq__
    cotangents: dict[Tensor, np.ndarray] = {root: seed}

    # reverse topological order guarantees a node's cotangent is complete before its rule runs
    for node in reversed(_topological_order(root)):
        g = cotangents.get(node)
        if g is None or node._vjp is None:  
            continue
        for inp, contribution in zip(node._inputs, node._vjp(g), strict=True):
            if contribution is None or not inp.requires_grad:
                continue
            c = _unbroadcast(np.asarray(contribution, dtype=np.float64), inp.data.shape)
            previous = cotangents.get(inp)
            cotangents[inp] = c.copy() if previous is None else previous + c

    return cotangents
