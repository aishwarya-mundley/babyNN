import numpy as np
from Parameter import Parameter

class Module(object):
  def __init__(self):
    self._parameters = {}  # Dictionary to hold parameters
    self._modules = {}     # Dictionary to hold sub-modules

  def __setattr__(self, name, value):
    if isinstance(value, Parameter):
      # print(f"  -> Identified as Parameter! Adding to _parameters['{name}']") # Debug print
      self._parameters[name] = value
      super().__setattr__(name, value)
    elif isinstance(value, Module):
      # print(f"  -> Identified as Module! Adding to _modules['{name}']") # Debug print
      self._modules[name] = value
      super().__setattr__(name, value)
    else:
      super().__setattr__(name, value)

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def forward(self, *args, **kwargs):
    raise NotImplementedError          # must be implemented by subclasses

  def parameters(self):
    # yields all the parameters of this module and all the sub-modules recursively
    for name, param in self._parameters.items():
      yield param
    for name, module in self._modules.items():
      yield from module.parameters()

  def zero_grad(self):
    # Iterate over the parameters by calling the parameters() method
    for p in self.parameters():
      p.grad = np.zeros_like(p.data, dtype=float)

class Linear(Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    limit = np.sqrt(1/in_features)
    # limit = np.sqrt(6.0 / (in_features + out_features)) # Xavier/Glorot initialization
    self.weight = Parameter(np.random.uniform(-limit, limit, (in_features, out_features)))
    self.bias = Parameter(np.random.uniform(0.0, 0.1, out_features))

  def forward(self, x):
    return x.matmul(self.weight) + self.bias
  
class ReLU(Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x.relu()
  
class Sigmoid(Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x.sigmoid()
  
class Identity(Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x
  
class Tanh(Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x.tanh()
  
class Sequential(Module):
  def __init__(self, *modules):
    super().__init__() # Initialize parent Module class
    for i, module in enumerate(modules):
      self._modules[str(i)] = module      # register sub-modules by index

  def forward(self, x):
    for module in self._modules.values():
      x = module(x)
    return x