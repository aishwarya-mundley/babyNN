import numpy as np
from Tensor import Tensor

class Categorical:
    def __init__(self, probs=None, logits=None):
        if probs is None and logits is None:
            raise ValueError("provide probs or logits")
        if probs is not None:
            arr = probs if isinstance(probs, Tensor) else Tensor(np.array(probs, dtype=np.float32))
            # Normalize using Tensor ops
            self.probs = arr / arr.sum(axis=-1, keepdims=True)
        else:
            logits = logits if isinstance(logits, Tensor) else Tensor(np.array(logits, dtype=np.float32))
            # softmax with Tensor ops
            maxl = logits - logits.data.max(axis=-1, keepdims=True)  # NOTE: numeric stable; if logits is Tensor use .data here only for shape; simpler: compute exp(logits - max)
            exp = (logits - logits.data.max(axis=-1, keepdims=True)).exp()
            self.probs = exp / exp.sum(axis=-1, keepdims=True)

    def sample(self):
        # sampling is not differentiable; return Tensor with requires_grad=False
        if self.probs.data.ndim == 1:
            choice = np.random.choice(self.probs.data.shape[-1], p=self.probs.data)
            return Tensor(np.array(choice, dtype=np.int64), requires_grad=False)
        else:
            # batch sample
            batch = self.probs.data.shape[0]
            choices = np.array([np.random.choice(self.probs.data.shape[-1], p=self.probs.data[i])
                                for i in range(batch)], dtype=np.int64)
            return Tensor(choices, requires_grad=False)

    def log_prob(self, value):
        # value is integer index or Tensor of ints (requires_grad=False)
        if not isinstance(value, Tensor):
            value = Tensor(np.array(value, dtype=np.int64), requires_grad=False)

        p = self.probs  # Tensor
        if p.data.ndim == 1:
            # one-hot of single index
            one_hot = np.eye(p.data.shape[-1], dtype=np.float32)[int(value.data)]
            one_hot_t = Tensor(one_hot, requires_grad=False)
            selected = (p * one_hot_t).sum(axis=-1)   # Tensor ops -> selected depends on p
            return selected.log()
        else:
            # batch: build one-hot matrix (batch, num_actions)
            oh = np.eye(p.data.shape[-1], dtype=np.float32)[value.data]  # value.data shape (batch,)
            one_hot_t = Tensor(oh, requires_grad=False)
            selected = (p * one_hot_t).sum(axis=-1)   # shape (batch,)
            return selected.log()

    def entropy(self):
        p = self.probs
        # -sum p log p (use Tensor ops so gradient flows)
        # add small epsilon for numeric stability
        eps = Tensor(1e-8, requires_grad=False)
        return - (p * (p + eps).log()).sum(axis=-1)
