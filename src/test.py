import os
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from model.encoder import ContrastiveEncoder
from model.classification import LinearEvaluator
from model.standardization import standardize_xray
from model.utils.load_config import cfg
from model.utils.dataloader import get_dataloader

if __name__ == "__main__":
    epoch = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Trained Model (from /weights)
    encoder = ContrastiveEncoder().to(device)
    encoder.load_state_dict(torch.load(f'src/weights/finetuned_encoder_epoch_{epoch}.pth', map_location=device))
    classifier = LinearEvaluator(feature_dim=encoder.feature_dim).to(device)
    classifier.load_state_dict(torch.load(f'src/weights/classifier_head_epoch_{epoch}.pth', map_location=device))
    encoder.eval()
    classifier.eval()

    # Load test dataset
    test_dataloader = get_dataloader("src/dataset/test")

    # Calculate final metrics
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for images, labels, _ in test_dataloader:
            h, _ = encoder(images.to(device))
            probs = torch.nn.functional.softmax(classifier(h), dim=1)[:, 1]
            
            all_targets.extend(labels.numpy())
            all_preds.extend((probs >= 0.5).int().cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    print(f"At: {epoch} Total: {len(test_dataloader)} Acc: {accuracy_score(all_targets, all_preds) * 100:.2f}% F1 (pneumo): {f1_score(all_targets, all_preds):.4f}")
    print(f"True Negatives (Normal): {cm[0][0]}  |  False Positives: {cm[0][1]}")
    print(f"False Negatives (Missed): {cm[1][0]}  |  True Positives (Pneumonia): {cm[1][1]}")