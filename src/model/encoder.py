import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ContrastiveEncoder(nn.Module):
    def __init__(self, projection_dim=128):
        """
        Compress high dimensional (cfg.img_size)**2 into latent representation / embedding
        """
        super().__init__()
        # Load standard ResNet50
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Isolate the backbone (remove the fully connected classification layer)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_dim = base_model.fc.in_features # 2048 vector for ResNet50
        
        # A 2-layer MLP with ReLU hidden activation (Projection Head)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, projection_dim)
        )

    def forward(self, x):
        # 1. Extract representation h
        h = self.backbone(x)
        h = torch.flatten(h, 1)
        
        # 2. Map to latent space z
        z = self.projection_head(h)
        
        # Normalize projections onto the unit hypersphere for Cosine Similarity
        z = nn.functional.normalize(z, dim=1)
        return h, z