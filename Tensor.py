import math
import numpy as np

NO_GRAD = False
class no_grad:
  def __enter__(self):
    global NO_GRAD
    self.prev = NO_GRAD
    NO_GRAD = True

  def __exit__(self, exc_type, exc_value, traceback):
    global NO_GRAD
    NO_GRAD = self.prev

class Tensor:
  def __init__(self, data, _children=(), _op='', requires_grad=True):
    global NO_GRAD
    # Allow Parameter objects to pass through (since Parameter inherits from Tensor)
    if hasattr(data, 'data') and hasattr(data, 'grad'):  # It's a Tensor-like object
      self.data = data.data if hasattr(data, 'data') else data
    elif isinstance(data, (int, float, np.ndarray, np.generic)):
      if isinstance(data, (int, float, np.generic)):
        self.data = np.array(data, dtype=np.float32)  # Convert to float32 for consistency
      else:
        self.data = data.astype(np.float32)  # Ensure numpy array is float32
    else:
      raise TypeError(f"Data must be a number or numpy array, got {type(data)}")
    self.grad = np.zeros_like(self.data, dtype=float)  #initialize gradiant
    self._backward = lambda:None
    self._op = _op
    self._prev = set(_children) # Set of input Tensors that created this Tensor
    self.is_parameter = False # Flag for identifying parameters
    self.is_leaf = len(_children) == 0
    self.requires_grad = requires_grad and not NO_GRAD # if true only then participate in backward pass

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
      grad_out = out.grad
      if axis is None:
        # Summed over all dims - just broadcast scalar to input shape
        self.grad = self.grad + np.ones_like(self.data) * grad_out
      else:
        # Ensure axis is tuple for consistent processing
        if isinstance(axis, int):
          axes = (axis,)
        else:
          axes = tuple(axis)
        
        # If keepdims=False, expand dimensions back for broadcasting
        if not keepdims:
          for ax in sorted(axes):
            grad_out = np.expand_dims(grad_out, ax)
        
        # Broadcasr to input shape
        self.grad = self.grad + np.ones_like(self.data) * grad_out
    out._backward = _backward
    return out
  
  def mean(self, axis=None, keepdims=False):
    out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=self.requires_grad, _op="mean")

    if self.requires_grad:
      def _backward():
        grad_shape = self.data.shape
        # Number of elements the mean is taken over
        if axis is None:
          num_elements = self.data.size
          grad = (out.grad / num_elements) * np.ones_like(self.data)
        else:
          # Normalize only along given axis
          num_elements = self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[i] for i in axis])
          grad = (out.grad / num_elements)

          if not keepdims:
            # Expand back to original shape
            grad = np.expand_dims(grad, axis=axis)

          grad = np.broadcast_to(grad, grad_shape)
        self.grad = self.grad + grad if self.grad is not None else grad
      out._backward = _backward
      out._prev = {self}
    return out

  def log(self):
    out = Tensor(np.log(self.data), (self,), 'log')

    def _backward():
      epsilon = 1e-8
      self.grad = self.grad + out.grad * (1.0 / (self.data + epsilon))
    out._backward = _backward
    return out

  def exp(self):
    out = Tensor(np.exp(self.data), (self,), 'exp')
    out.requires_grad = self.requires_grad
    
    def _backward():
        if self.requires_grad:
            self.grad = self.grad + out.grad * out.data  # d/dx(e^x) = e^x
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
  
  def squeeze(self, axis=None):
    return Tensor(np.squeeze(self.data, axis=axis))
  
  def numpy(self):
    # Return the raw NumPy array without breaking references
    return np.array(self.data, copy=False)
  
  def item(self):
    if self.data.size != 1:
      raise ValueError("Can only convert a tensor with one element to a Python scalar")
    return self.data.item()

  def __repr__(self):
    return f"Tensor(data={self.data}, grad={self.grad})"