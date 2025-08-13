import numpy as np
from Tensor import Tensor

class Normal:
    def __init__(self, mean, std):
        # Accept Tensor or convertible objects; store as Tensor
        self.mean = mean if isinstance(mean, Tensor) else Tensor(np.array(mean, dtype=np.float32), requires_grad=False)
        self.std  = std  if isinstance(std,  Tensor) else Tensor(np.array(std, dtype=np.float32),  requires_grad=False)

    def sample(self):
        eps = np.random.randn(*self.mean.data.shape).astype(np.float32)
        return Tensor(self.mean.data + eps * self.std.data, requires_grad=False)

    def log_prob(self, value):
        # Ensure value is a Tensor (value is observation/action, doesn't need grad)
        if not isinstance(value, Tensor):
            value = Tensor(np.array(value, dtype=np.float32), requires_grad=False)

        # Use Tensor ops so graph links to mean/std (which themselves may come from network Tensors)
        # var = std ** 2
        var = self.std * self.std
        log_scale = self.std.log()
        # term = - (value - mean)^2 / (2*var)
        term = ((value - self.mean) ** 2) * ( -0.5 / var )
        const = Tensor(np.log(np.sqrt(2.0 * np.pi)), requires_grad=False)
        return term - log_scale - const  # this is a Tensor built from Tensor ops

    def entropy(self):
        # 0.5 * log(2*pi*e*sigma^2)
        # Use Tensor ops: create constant as Tensor (no grad)
        const = Tensor(0.5 * np.log(2.0 * np.pi * np.e), requires_grad=False)
        return const + (self.std.log())  # since 0.5*log(sigma^2) = log(sigma)

    def __repr__(self):
        return f"Normal(mean={self.mean}, std={self.std})"