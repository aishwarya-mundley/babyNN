import numpy as np
from blocks.Module import Sequential, Linear, ReLU
from blocks.Optimizer import Adam
from blocks.Tensor import Tensor

def mse_loss(predictions, target):
  return (predictions - target)**2.0

# Training data
X_train = Tensor(np.array([[1.0], [2.0], [3.0], [4.0]]))
y_train = Tensor(np.array([[2.0], [4.0], [6.0], [8.0]]))

# Model definition
model = Sequential(Linear(in_features=1, out_features=10), ReLU(), Linear(in_features=10, out_features=1))

# Optimizer
# optimizer = SGD(model.parameters(), lr=0.01)
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
epochs=100
print("Starting training..")
for epoch in range(epochs):
  # forward pass
  predictions = model(X_train)

  # calculate loss
  loss = mse_loss(predictions, y_train).sum()

  # zero grad
  optimizer.zero_grad()

  # backward pass
  loss.backward()

  # update parameters
  optimizer.step()

  if (epoch + 1) % 10 == 0:
    print(f"Epoch {epoch+1}/{epochs}, Loss {loss.data.item():.4f}")

print("\nTraining complete!")
print("Final predictions:", predictions.data)
print("Final loss:", loss.data.item())
print("Learned parameters:")
for name, param in model._parameters.items():
  print(f"  {name}: {param.data}")
for name, sub_module in model._modules.items():
  if isinstance(sub_module, Linear):
    print(f"  Linear Layer {name} weights: {sub_module.weight.data}")
    print(f"  Linear Layer {name} biases: {sub_module.bias.data}")