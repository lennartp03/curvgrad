import numpy as np

from curvgrad.tensor import Tensor


def cross_entropy_loss(logits: Tensor, y_onehot: np.ndarray) -> Tensor:
    """Return the softmax cross entropy averaged over the batch. Both inputs have shape N by K."""
    return -(logits.log_softmax() * Tensor(y_onehot)).sum(axis=-1).mean()
