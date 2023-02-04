import numpy as np
import torch
import matplotlib.pyplot as plt

# Generate some synthetic data
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = 2 * x + 3 + 0.1 * np.random.randn(100)

# Convert the data to tensors
x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Define the neural network
class RegressionModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# Initialize the model
input_size = 1
output_size = 1
model = RegressionModel(input_size, output_size)

# Define the loss function and the optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot the data and the model's prediction
plt.scatter(x.numpy(), y.numpy(), c='blue')
plt.plot(x.numpy(), y_pred.detach().numpy(), c='red')
plt.show()
