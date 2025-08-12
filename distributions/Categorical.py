import numpy as np
from Tensor import Tensor

class Categorical:
    def __init__(self, probs=None, logits=None):
        if probs is None and logits is None:
            raise ValueError("Must provide probs or logits")

        if probs is not None:
            arr = probs.data if isinstance(probs, Tensor) else np.array(probs, dtype=np.float32)
            arr = arr / arr.sum(axis=-1, keepdims=True)  # normalize
            self.probs = Tensor(arr)
        else:
            arr = logits.data if isinstance(logits, Tensor) else np.array(logits, dtype=np.float32)
            # Softmax
            exp_logits = np.exp(arr - np.max(arr, axis=-1, keepdims=True))
            arr = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
            self.probs = Tensor(arr)

    def sample(self):
        # Works for batched and unbatched
        if self.probs.data.ndim == 1:
            choice = np.random.choice(len(self.probs.data), p=self.probs.data)
            return Tensor(choice)
        else:
            choices = [np.random.choice(len(p), p=p) for p in self.probs.data]
            return Tensor(np.array(choices, dtype=np.int64))

    def log_prob(self, value):
        value = value if isinstance(value, Tensor) else Tensor(np.array(value, dtype=np.int64))
        if self.probs.data.ndim == 1:
            return Tensor(np.log(self.probs.data[value.data]))
        else:
            return Tensor(np.log(self.probs.data[np.arange(len(value.data)), value.data]))

    def entropy(self):
        p = self.probs.data
        return Tensor(-np.sum(p * np.log(p + 1e-8), axis=-1))
    
    def __repr__(self):
        return f"Categorical(probs={self.probs})"
