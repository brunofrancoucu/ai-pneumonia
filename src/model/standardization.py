import cv2
import numpy as np
from pathlib import Path

def standardize_xray(image_path: Path | str, target_size, zoom_factor) -> np.ndarray:
    """
    Executes offline standardization:
    1. Center-crop to a square.
    2. Scale by zoom_factor.
    3. Resize to target dimensions.
    4. Apply CLAHE.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if (img is None): raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # 1. Crop to center square
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    square_img = img[start_y:start_y+min_dim, start_x:start_x+min_dim]
    
    # 2. Apply Zoom Factor
    zoom_dim = int(min_dim / zoom_factor)
    zoom_start = (min_dim - zoom_dim) // 2
    zoomed_img = square_img[zoom_start : zoom_start + zoom_dim, 
                            zoom_start : zoom_start + zoom_dim]
    
    # 3. Resize to target size (256x256)
    resized_img = cv2.resize(zoomed_img, target_size, interpolation=cv2.INTER_CUBIC)
    
    # 4. Illumination & Contrast: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    standardized_img = clahe.apply(resized_img)
        
    return standardized_img