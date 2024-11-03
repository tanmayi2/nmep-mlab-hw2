import torch
import torch.nn as nn

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

            #nn.AdaptiveAvgPool2d((6, 6)),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # # Calculate the size of the flattened features
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
        # Dummy forward pass to calculate the size of the flattened features
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