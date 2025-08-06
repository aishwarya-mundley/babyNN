import math
import numpy as np

class Tensor:
  def __init__(self, data, _children=(), _op='', requires_grad=True):
    # Allow Parameter objects to pass through (since Parameter inherits from Tensor)
    if hasattr(data, 'data') and hasattr(data, 'grad'):  # It's a Tensor-like object
      self.data = data.data if hasattr(data, 'data') else data
    elif isinstance(data, (int, float, np.ndarray)):
      if isinstance(data, (int, float)):
        self.data = np.array(data)
      else:
        self.data = data
    else:
      raise TypeError(f"Data must be a number or numpy array, got {type(data)}")
    self.grad = np.zeros_like(self.data, dtype=float)  #initialize gradiant
    self._backward = lambda:None
    self._op = _op
    self._prev = set(_children) # Set of input Tensors that created this Tensor
    self.is_parameter = False # Flag for identifying parameters
    self.is_leaf = len(_children) == 0
    self.requires_grad = requires_grad  # if true only then participate in backward pass

  def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad = self.grad + out.grad
            other.grad = other.grad + out.grad
        out._backward = _backward
        return out

  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad = self.grad + other.data * out.grad
      other.grad = other.grad + self.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Tensor(self.data**other, (self,), f'**{other}')

    def _backward():
      self.grad = self.grad + (other * self.data**(other-1)) * out.grad
    out._backward = _backward
    return out

  def relu(self):
    out = Tensor(np.maximum(0, self.data), (self,), 'relu')

    def _backward():
      self.grad = self.grad + (out.data > 0) * out.grad
    out._backward = _backward
    return out
  
  def sigmoid(self):
    out = Tensor(1/(1 + math.exp(-self.data)), (self,), 'sigmoid')

    def _backward():
      self.grad = self.grad + (out.data * (1 - out.data)) * out.grad
    out._backward = _backward
    return out

  # Basic matrix multiplication for neural networks
  def matmul(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    assert self.data.ndim == 2 and other.data.ndim == 2 and self.data.shape[1] == other.data.shape[0], f"Shape mismatch for matmul: {self.data.shape} @ {other.data.shape}"

    out = Tensor(self.data @ other.data, (self, other), '@')

    def _backward():
      self.grad = self.grad + out.grad @ other.data.T
      other.grad = other.grad + self.data.T @ out.grad
    out._backward = _backward
    return out

  def sum(self, axis=None, keepdims=False):
    out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')

    def _backward():
      if axis is not None and not keepdims:
        # Need to expand grad if sum reduced dimensions
        shape_tuple = tuple(1 if i == axis else self.data.shape[i] for i in range(self.data.ndim))
        self.grad = self.grad + np.reshape(out.grad, shape_tuple)
      else:
        self.grad = self.grad + out.grad
    out._backward = _backward
    return out

  def log(self):
    out = Tensor(np.log(self.data), (self,), 'log')

    def _backward():
      epsilon = 1e-8
      self.grad = self.grad + out.grad * (1.0 / (self.data + epsilon))
    out._backward = _backward
    return out

  def tanh(self):
    out = Tensor(np.tanh(self.data), (self,), 'tanh')

    def _backward():
      tanh_val = np.tanh(self.data)
      self.grad = self.grad + out.grad * (1 - tanh_val**2)
    out._backward = _backward
    return out

  def backward(self):
    # Topological sort for correct backpropagation order
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = np.ones_like(self.data, dtype=float) # Initialize gradient for the output
    for node in reversed(topo):
      node._backward()

  # Enable operations like -x, x-y, /x, x/y
  def __neg__(self): # -self
    return self * -1

  def __sub__(self, other): # self - other
    return self + (-other)

  def __rsub__(self, other): # other - self
    return other + (-self)

  def __truediv__(self, other): # self / other
    return self * (other**-1)

  def __rtruediv__(self, other): # other / self
    return other * (self**-1)

  def __repr__(self):
    return f"Tensor(data={self.data}, grad={self.grad})"