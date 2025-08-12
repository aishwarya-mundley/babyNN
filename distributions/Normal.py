import numpy as np
from Tensor import Tensor

class Normal:
    def __init__(self, mean, std):
        # Convert everything to Tensor
        self.mean = mean if isinstance(mean, Tensor) else Tensor(np.array(mean, dtype=np.float32))
        self.std = std if isinstance(std, Tensor) else Tensor(np.array(std, dtype=np.float32))

    def sample(self):
        eps = np.random.randn(*self.mean.data.shape).astype(np.float32)
        return Tensor(self.mean.data + eps * self.std.data)

    def log_prob(self, value):
        value = value if isinstance(value, Tensor) else Tensor(np.array(value, dtype=np.float32))
        var = self.std.data ** 2
        log_scale = np.log(self.std.data)
        return Tensor(-((value.data - self.mean.data) ** 2) / (2 * var) - log_scale - np.log(np.sqrt(2 * np.pi)))

    def entropy(self):
        # Formula: 0.5 * log(2πeσ²)
        return Tensor(0.5 * np.log(2 * np.pi * np.e * (self.std.data ** 2)))
    
    def __repr__(self):
        return f"Normal(mean={self.mean}, std={self.std})"
