# This class is the data-container abstraction: Tensor, as predefined in PyTorch or related
# libraries. It is initialized with some data array and stores the variables required for the
# backward pass of the model. In the backward pass this tensor is one node that resulted from some
# operation (see ops.py), and for each forward pass we store the following.
#
# Assume some linear layer y = Wx where W is shape (o, n) and x is (n, d):
#   self.requires_grad : whether autograd.vjp computes a cotangent for this node. Constants such as
#                  the 3.0 in x * 3.0 do not need one. Gradients are NOT stored on the tensor:
#                  autograd.vjp returns them as a dict mapping each node to dL/d(node).
#   self._vjp    : the LOCAL rule of the op that produced this node. It takes g = dL/d(out) and
#                  returns dL/d(input_i) = g^T (d out / d input_i) for each input, i.e. it applies
#                  the transposed local Jacobian. It knows ONE op and nothing about the graph --
#                  composing the rules along the graph is autograd.py's job, not the rule's.
#   self._inputs : the nodes this one was built from, ORDERED so that they line up positionally
#                  with what _vjp returns (a set would lose both the order and, for x + x, an edge).
#   self._jvp    : the forward-mode counterpart.
#   self._op     : label of the producing op, for printing / dumping the graph.
#
# Returned as a static method for all objects of the class is _ensure_tensor, plus the common atomic
# mathematical operations and the standard activation functions and reshape used in neural networks.
#
# Speaking from a graphical point of view those tensors are the nodes in a compute graph, resulting
# from atomic operations on previous nodes, which we can split up and traverse backwards and forward.

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from curvgrad import ops


class Tensor:
    _inputs: tuple[Tensor, ...]
    _vjp: Callable[[np.ndarray], tuple[np.ndarray | None, ...]] | None
    _jvp: Callable[..., np.ndarray] | None
    _op: str

    __array_ufunc__ = None

    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self._inputs = ()
        self._vjp = None
        self._jvp = None
        self._op = "leaf"

    @property
    def shape(self):
        return np.shape(self.data)

    def __repr__(self):
        return f"Tensor({self.data!r}, op={self._op!r}, requires_grad={self.requires_grad})"

    @staticmethod
    def _ensure_tensor(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    def __add__(self, other):
        return ops.add(self, self._ensure_tensor(other))

    def __radd__(self, other):
        return ops.add(self._ensure_tensor(other), self)

    def __sub__(self, other):
        return ops.sub(self, self._ensure_tensor(other))

    def __rsub__(self, other):
        return ops.sub(self._ensure_tensor(other), self)

    def __mul__(self, other):
        return ops.mul(self, self._ensure_tensor(other))

    def __rmul__(self, other):
        return ops.mul(self._ensure_tensor(other), self)

    def __truediv__(self, other):
        return ops.div(self, self._ensure_tensor(other))

    def __rtruediv__(self, other):
        return ops.div(self._ensure_tensor(other), self)

    def __matmul__(self, other):
        return ops.matmul(self, self._ensure_tensor(other))

    def __rmatmul__(self, other):
        return ops.matmul(self._ensure_tensor(other), self)

    def __neg__(self):
        return ops.neg(self)

    def sum(self, axis=None):
        return ops.sum_(self, axis)

    def mean(self, axis=None):
        return ops.mean_(self, axis)

    def reshape(self, *shape):
        # accept both t.reshape(2, 3) and t.reshape((2, 3))
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return ops.reshape(self, shape)

    def relu(self):
        return ops.relu(self)

    def sigmoid(self):
        return ops.sigmoid(self)

    def softmax(self):
        return ops.softmax(self)

    def exp(self):
        return ops.exp(self)

    def log(self):
        return ops.log(self)

    def log_softmax(self):
        return ops.log_softmax(self)

    def abs(self):
        return ops.abs_(self)

    def __pow__(self, other):
        return ops.pow_(self, other)

    @property
    def T(self):
        return ops.transpose(self)
    
