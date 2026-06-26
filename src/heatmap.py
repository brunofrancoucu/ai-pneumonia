import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model.encoder import ContrastiveEncoder
from model.classification import LinearEvaluator
from model.standardization import standardize_xray

# 1. The Glue: Combine your two models into one seamless network
class PneumoniaNetwork(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        
    def forward(self, x):
        # Your encoder returns (h, z). We only want h for the classifier.
        h, _ = self.encoder(x)
        return self.classifier(h)

def generate_heatmap(img_path, save_path, device):
    print("--- Loading Architectures ---")
    encoder = ContrastiveEncoder().to(device)
    classifier = LinearEvaluator(feature_dim=encoder.feature_dim).to(device)
    
    # Load your Phase 3 Fine-Tuned weights!
    encoder.load_state_dict(torch.load('src/weights/finetuned_encoder_epoch_10.pth', map_location=device))
    classifier.load_state_dict(torch.load('src/weights/classifier_head_epoch_10.pth', map_location=device))
    
    # Ensure they are in eval mode
    encoder.eval()
    classifier.eval()

    # Wrap them together
    model = PneumoniaNetwork(encoder, classifier).to(device)

    # 2. Target the last convolutional layer of your ResNet backbone
    # (Assuming your ResNet is initialized as self.backbone inside ContrastiveEncoder)
    print(model.encoder.backbone)
    target_layers = [model.encoder.backbone[7][-1]] # 2048-filter convolutions (the Bottleneck layers)

    # 3. Prepare the Image using your exact standardization pipeline
    clahe_img = standardize_xray(img_path, target_size=(256, 256), zoom_factor=1)
    img_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    
    # Grad-CAM requires the visual overlay to be a float32 array between 0 and 1
    rgb_img_float = np.float32(img_rgb) / 255.0
    
    # Apply your Phase 2 mathematical tensor formatting
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img_rgb).unsqueeze(0).to(device) # Add batch dimension

    # 4. Generate the Heatmap
    # Target class 1 (Pneumonia)
    targets = [ClassifierOutputTarget(1)] 
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # Get the heatmap matrix
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Overlay the heatmap (red/yellow) onto your black-and-white X-ray
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    # 5. Save the result
    # Convert RGB back to BGR for OpenCV saving
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM heatmap to {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Point this at an image you know has Pneumonia
    filename = "pneumo_1465v2537"
    test_image_path = f"src/dataset/test/{filename}.jpeg" 
    output_path = f"output/heatmap_{filename}.jpg"
    
    generate_heatmap(test_image_path, output_path, device)