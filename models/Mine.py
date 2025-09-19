# models/mine.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    A standard Residual Block used in ResNet.
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MINE(nn.Module):
    """
    Mutual Information Neural Estimator (MINE) network.
    It takes two inputs (input data and feature data) and outputs a scalar score.
    """

    def __init__(self, block, num_blocks, num_classes=10):
        super(MINE, self).__init__()
        self.in_planes = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # Output: batch x 512 x 4 x 4

        # Fully connected layers
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Build a sequence of residual blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input, feature):
        """
        Forward pass of MINE.
        Args:
            input: First input tensor (B, 3, H, W)
            feature: Second input tensor (B, C, H', W')
        Returns:
            Scalar MI score for each pair in the batch.
        """
        # Process input image
        out = F.relu(self.bn1(self.conv1(input)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)

        # Process feature map
        feature = F.avg_pool2d(feature, 4)

        # Flatten and pass through FC layers
        input_out = self.linear1(out.view(out.size(0), -1))
        input_out = self.linear2(input_out)

        feature_out = self.linear1(feature.view(feature.size(0), -1))
        feature_out = self.linear2(feature_out)

        # Combine results
        combined = input_out + feature_out
        return combined
