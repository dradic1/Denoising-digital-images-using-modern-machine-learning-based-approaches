import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        )
        
        layers = []
        for _ in range(num_of_layers - 2):
            layers.append(ResidualBlock(features))
        self.body = nn.Sequential(*layers)
        
        self.tail = nn.Conv2d(in_channels=channels + 2 * features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        out_head = self.head(x)
        out_body = self.body(out_head)
        combined = torch.cat((x, out_head, out_body), dim=1)
        out = self.tail(combined)
        return out
