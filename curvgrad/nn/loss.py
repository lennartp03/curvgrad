import numpy as np

from curvgrad.tensor import Tensor


def cross_entropy_loss(logits: Tensor, y: np.ndarray) -> Tensor:
    """Softmax cross entropy, averaged over the batch."""
    return -(logits.log_softmax() * Tensor(y)).sum(axis=-1).mean()


def cross_entropy_hvp(logits: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Cross entropy Hessian with respect to the logits, applied to u.

    Per sample (diag(p) - p p^T) u / N. Independent of the labels.
    """
    shift = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(shift)
    p = e / np.sum(e, axis=-1, keepdims=True)
    hu = p * u - p * np.sum(p * u, axis=-1, keepdims=True)
    return hu / logits.shape[0]
