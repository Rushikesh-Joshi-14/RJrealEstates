import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

# Parameters
input_size = 3
output_size = 1

# Create model instance
model = SimpleModel(input_size, output_size)

# Sample input
sample_input = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

# Forward pass
output = model(sample_input)

print("Input:", sample_input)
print("Output:", output)

# Define a simple loss
target = torch.tensor([[1.0]])  # Expected output
criterion = nn.MSELoss()  # Mean Squared Error loss
loss = criterion(output, target)

print("Loss:", loss.item())

# Backward pass and optimization
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()  # Clear gradients
loss.backward()  # Compute gradients
optimizer.step()  # Update weights

print("Updated weights:", list(model.parameters()))
