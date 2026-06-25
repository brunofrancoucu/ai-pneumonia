import torch.nn as nn

class LinearEvaluator(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=2):
        """
        Classification Head: Converts 1D array 2048 Dimensions => [2]
        """
        super().__init__()
        # A single linear transformation layer
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, h):
        return self.classifier(h)