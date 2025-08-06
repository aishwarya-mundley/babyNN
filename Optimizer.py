import numpy as np

class SGD():
  def __init__(self, parameters, lr):
    # Collect parameters from the generator into a list
    self.parameters = list(parameters)
    self.lr = lr

  def step(self):
    for p in self.parameters:
      # Ensure parameter has a gradient before updating
      if p.grad is not None:
        p.data = p.data - self.lr * p.grad

  def zero_grad(self):
    for p in self.parameters:
      p.grad = np.zeros_like(p.data, dtype=float)

class Adam():
  def __init__(self, parameters, lr):
    self.parameters = list(parameters)
    self.lr = lr
    self.beta1 = 0.9
    self.beta2 = 0.999
    self.epsilon = 1e-8
    # Initialize m and v as lists of zero arrays, one for each parameter
    self.m = [np.zeros_like(p.data, dtype=float) for p in self.parameters]
    self.v = [np.zeros_like(p.data, dtype=float) for p in self.parameters]
    self.t = 0

  def step(self):
    self.t += 1
    for i, p in enumerate(self.parameters):
      if p.grad is not None:
        # Update biased first and second moment estimates
        self.m[i] = self.beta1 * self.m[i] + ((1 - self.beta1) * p.grad)              # similar like SGD with momentum (not squared)
        self.v[i] = self.beta2 * self.v[i] + ((1 - self.beta2) * (p.grad * p.grad))   # similar like RMSProp (squared)

        # Compute bias-corrected first and second moment estimates
        m_hat = self.m[i] / (1 - (self.beta1 ** self.t))
        v_hat = self.v[i] / (1 - (self.beta2 ** self.t))

        # Update parameters
        p.data = p.data - (self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon)))

  def zero_grad(self):
    for p in self.parameters:
      p.grad = np.zeros_like(p.data, dtype=float)