from Tensor import Tensor

class Parameter(Tensor):
  def __init__(self, data):
    super().__init__(data)
    self.is_parameter = True