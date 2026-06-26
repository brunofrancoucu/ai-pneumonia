import os
import torch
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from model.encoder import ContrastiveEncoder
from model.classification import LinearEvaluator
from model.standardization import standardize_xray
from model.utils.load_config import cfg
from model.utils.dataloader import get_dataloader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = 30
    
    # Load ResNet
    encoder = ContrastiveEncoder().to(device)
    encoder.load_state_dict(torch.load(f'src/weights/finetuned_encoder_epoch_{epoch}.pth', map_location=device))
    # Load Classifier Head
    class_head = LinearEvaluator(feature_dim=encoder.feature_dim).to(device)
    class_head.load_state_dict(torch.load(f'src/weights/classifier_head_epoch_{epoch}.pth', map_location=device))
    encoder.eval()
    class_head.eval()

    # Load Validation Data
    loader = get_dataloader("src/dataset/__vali2")

    # 2. Run Inference
    results = {'Id': [], 'Label': []}
    with torch.no_grad():
        for images, labels, img_ids in loader:
            h, _ = encoder(images.to(device))
            probs = torch.nn.functional.softmax(class_head(h), dim=1)[:, 1]
            
            results['Id'].extend(img_ids)
            results['Label'].extend((probs >= 0.5).int().cpu().numpy())

    # 3. Export exact Kaggle format
    pd.DataFrame(results).to_csv(f'src/validation_prediction_epoch_{epoch}.csv', index=False)
    print("Saved to src/validation_prediction.csv")