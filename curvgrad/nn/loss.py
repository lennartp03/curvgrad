import numpy as np

from curvgrad.tensor import Tensor


def cross_entropy_loss(logits: Tensor, y: np.ndarray) -> Tensor:
    """Return the softmax cross entropy averaged over the batch. Both inputs have shape N by K."""
    return -(logits.log_softmax() * Tensor(y)).sum(axis=-1).mean()


def cross_entropy_hvp(logits: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Return H u where H is the Hessian of cross_entropy_loss with respect to the logits.

    For one sample with probabilities p this is p * u - p * sum(p * u). The loss averages over the
    batch, so divide by N. Labels do not appear because this Hessian does not depend on them.
    Both arguments have shape N by K and are plain arrays, since this is curvature and not a graph op.
    """
    raise NotImplementedError
