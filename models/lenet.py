import torch
from torch import nn
import gpustat


class LeNet(nn.Module):
    """Fake LeNet with 32x32 color images and 200 classes"""

    def __init__(self, num_classes: int = 200) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# AlexNet model definition
class AlexNet(nn.Module):
    def __init__(self, num_classes=10, img_size=70):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.feature_size = self._get_conv_output(img_size)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def _get_conv_output(self, img_size):
        dummy_input = torch.zeros(1, 3, img_size, img_size)
        output = self.features(dummy_input)
        output = self.avgpool(output)
        return int(torch.prod(torch.tensor(output.size())))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 4.1 
# Function to calculate parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize models
lenet = LeNet(num_classes=200)
alexnet = AlexNet(num_classes=10, img_size=70)

# Count parameters
lenet_params = count_parameters(lenet)
alexnet_params = count_parameters(alexnet)

print(f"LeNet Parameters: {lenet_params}")
print(f"AlexNet Parameters: {alexnet_params}")

# Define batch size and input size
batch_size = 32
input_size_lenet = (batch_size, 3, 32, 32)
input_size_alexnet = (batch_size, 3, 70, 70)

# Estimate GPU memory usage
def estimate_gpu_memory(model, input_size):
    input_tensor = torch.randn(*input_size).cuda()
    model = model.cuda()
    with torch.no_grad():
        _ = model(input_tensor)
    torch.cuda.empty_cache()
    gpu_usage = gpustat.new_query().jsonify()["gpus"][0]["memory.used"]
    return gpu_usage

# Initialize CUDA and measure memory usage
print("Measuring GPU memory usage for each model with gpustat...")

# Measure LeNet memory usage
lenet_memory = estimate_gpu_memory(lenet, input_size_lenet)
print(f"LeNet Memory Usage: {lenet_memory} MiB")

# Measure AlexNet memory usage
alexnet_memory = estimate_gpu_memory(alexnet, input_size_alexnet)
print(f"AlexNet Memory Usage: {alexnet_memory} MiB")

# Results summary
print("\nSummary:")
print(f"LeNet Parameters: {lenet_params}")
print(f"AlexNet Parameters: {alexnet_params}")
print(f"LeNet Memory Usage: {lenet_memory} MiB")
print(f"AlexNet Memory Usage: {alexnet_memory} MiB")