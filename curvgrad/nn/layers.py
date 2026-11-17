from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from curvgrad.nn.module import Module
from curvgrad.tensor import Tensor


class Linear(Module):
    """Affine layer y = x @ W + b with W of shape dim_in by dim_out and b of shape dim_out."""

    def __init__(self, dim_in: int, dim_out: int, init: str = "xavier") -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.W = Tensor(self._initialize_weights(init), requires_grad=True)
        self.b = Tensor(np.zeros(dim_out), requires_grad=True)

    def _initialize_weights(self, init: str = "xavier") -> np.ndarray:
        """Draw W from a Xavier uniform or Kaiming normal distribution with the global RNG."""
        if init == "xavier":
            # Xavier Linear (Tanh/Linear)
            bound = np.sqrt(6 / (self.dim_in + self.dim_out))
            return np.random.uniform(-bound, bound, (self.dim_in, self.dim_out))
        elif init == "kaiming":
            # Kaiming Normal (ReLU)
            sigma = np.sqrt(2 / self.dim_in)
            return np.random.normal(0.0, sigma, (self.dim_in, self.dim_out))
        else:
            raise NotImplementedError(
                f"Initialization {init} has not yet been implemented. Choose xavier or kaiming."
            )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, solver: str = "LS") -> np.ndarray:
        """Return closed form least squares weights without intercept. Does not set W."""
        # Playground for different solvers
        if solver == "LS":
            return np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
        elif solver == "Cho":
            raise NotImplementedError("Cholesky solver is not implemented yet.")
        else:
            raise ValueError(f"Unknown solver {solver!r}. Choose LS or Cho.")

    def forward(self, x: Tensor) -> Tensor:
        """Return x @ W + b. The bias broadcasts over the batch axis."""
        if len(x.shape) != 2 or x.shape[-1] != self.dim_in:
            raise ValueError(f"expected a 2D input with {self.dim_in} columns, got shape {x.shape}")
        return x @ self.W + self.b

    def parameters(self) -> list[Tensor]:
        """Return W and b in this order."""
        return [self.W, self.b]


class MLP(Module):
    """Linear layers with relu or sigmoid between them and no activation after the last."""

    def __init__(self, sizes: Sequence[int], activation: str = "relu") -> None:
        super().__init__()
        if len(sizes) < 2:
            raise ValueError(f"need at least an input and an output size, got {list(sizes)}")
        if activation not in ("relu", "sigmoid"):
            raise ValueError(f"Unknown activation {activation!r}. Choose relu or sigmoid.")
        self.activation = activation
        init = "kaiming" if activation == "relu" else "xavier"
        self.layers = [Linear(a, b, init=init) for a, b in pairwise(sizes)]

    def forward(self, x: Tensor) -> Tensor:
        """Apply the layers in order, with the activation after every layer but the last."""
        *hidden, last = self.layers
        for layer in hidden:
            h = layer(x)
            x = h.relu() if self.activation == "relu" else h.sigmoid()
        return last(x)

    def parameters(self) -> list[Tensor]:
        """Return W and b of every layer, first layer first."""
        return [p for layer in self.layers for p in layer.parameters()]
