import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Net class as you provided
'''class Net(nn.Module):
    def __init__(self, l1, l2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)  # Assuming CLASSES = 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x'''

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize and export the model
'''model = Net(l1=128, l2=64)
dummy_input = torch.randn(1, 1, 32, 32)  # Batch size of 1, 1 channel, 32x32 input size
output = model(dummy_input)
print(output.shape)
torch.onnx.export(model, dummy_input, "net.onnx", input_names=["input"], output_names=["output"])'''


model = SimpleCNN(num_classes=10)
# Switch model to evaluation mode
model.eval()

# Dummy input for ONNX export (batch size = 1, channels = 1, height = 28, width = 28)
dummy_input = torch.randn(1, 1, 28, 28)

# Export the model to ONNX
onnx_file_path = "simple_cnn.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,
    input_names=["input"],            # Name of the input node
    output_names=["output"],          # Name of the output node
    dynamic_axes={
        "input": {0: "batch_size"},   # Allow variable batch size
        "output": {0: "batch_size"}   # Allow variable batch size
    },
    opset_version=11                  # ONNX opset version
)

print(f"Model successfully exported to {onnx_file_path}")