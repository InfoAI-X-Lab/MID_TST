import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# LeNet5 Feature Extractor
# -------------------------------
class LeNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)   # 3×32×32 → 6×32×32
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)                 # → 6×16×16
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)             # → 16×12×12
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)                 # → 16×6×6
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)           # → 120×2×2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))  # B × 120 × 2 × 2
        return x


# -------------------------------
# LeNet5 Classifier Head
# -------------------------------
class LeNetClassifier(nn.Module):
    def __init__(self, in_channels=120, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 2 * 2, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)     # Flatten: B × (C×H×W)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -------------------------------
# LeNet5 Full Model
# -------------------------------
class LeNetFullModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.extractor = LeNetFeatureExtractor()
        self.classifier = LeNetClassifier(num_classes=num_classes)

    def forward(self, x, return_features=False):
        feat = self.extractor(x)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits


# -------------------------------
# Factory Function
# -------------------------------
def lenet5(num_classes=10):
    return LeNetFullModel(num_classes=num_classes)
