import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Basic residual block
# -------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# -------------------------------
# Feature extractor
# -------------------------------
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_planes = 64

        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # B × 512 × 4 × 4
        return x

# -------------------------------
# Classification head
# -------------------------------
class ResNetClassifier(nn.Module):
    def __init__(self, in_channels=512, num_classes=10):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)                # B × C × 1 × 1
        x = x.view(x.size(0), -1)       # B × C
        return self.fc(x)               # B × num_classes

# -------------------------------
# ResNet full model (backbone + head)
# -------------------------------
class ResNetFullModel(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.extractor = ResNetFeatureExtractor(block, layers)
        self.classifier = ResNetClassifier(in_channels=512 * block.expansion, num_classes=num_classes)

    def forward(self, x, return_features=False):
        feat = self.extractor(x)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits

# -------------------------------
# Factory function
# -------------------------------
def resnet18(num_classes=10):
    return ResNetFullModel(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
