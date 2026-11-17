from abc import ABC, abstractmethod

from curvgrad.tensor import Tensor


class Module(ABC):
    """Base class for layers. Owns the parameters and builds graph nodes in forward."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Map x to the output by composing Tensor ops. Stores no state."""

    def __call__(self, x: Tensor) -> Tensor:
        """Call the module like a function. Alias for forward."""
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        """Return the learnable leaf Tensors. Their order defines the flat vector layout."""
        return []
