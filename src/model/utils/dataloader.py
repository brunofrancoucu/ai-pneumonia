# Dataset
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from model.standardization import standardize_xray
from model.utils.load_config import cfg
from PIL import Image as ImagePil
import numpy as np

class FormattedDataset(Dataset):
    def __init__(self, img_dir, transformations=[], num_views=1):
        self.img_dir = img_dir
        self.num_views = num_views
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.jpeg', '.JPEG'))]

        # Formatting/normalization from image to matrix
        self.transform = v2.Compose(transformations + [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((cfg["img_size"], cfg["img_size"])),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        file_name = self.images[idx]
        img_path = os.path.join(self.img_dir, file_name)
        label = 0 if file_name.startswith('normal_') else 1 # TODO: raise on 'unknown_' in /validation
        img_id = os.path.splitext(file_name)[0] # Extracts name without extension

        # Standardize directly from the file path
        img_rgb = cv2.cvtColor(standardize_xray(img_path, target_size=(cfg["img_size"], cfg["img_size"]), zoom_factor=1), cv2.COLOR_GRAY2RGB)

        if isinstance(img_rgb, np.ndarray): img_rgb = ImagePil.fromarray(img_rgb)
                    
        views = [self.transform(img_rgb) for _ in range(self.num_views)] if self.num_views > 1 else self.transform(img_rgb)
        return views, label, img_id

def get_dataloader(img_dir, transforms=[], num_views=1, shuffle=False):
    """
    Load images in `path`
    """
    dataset = FormattedDataset(img_dir, transforms, num_views)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=cfg["num_workers"], pin_memory=True)