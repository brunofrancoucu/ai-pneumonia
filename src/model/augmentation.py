from torchvision.transforms import v2
from model.utils.dataloader import get_dataloader
from model.utils.load_config import cfg

def augmented_dataloader(path: str):
    """
    Apply augmentation to images in `path`
    Returns tuple of transformed images with label 
    """
    return get_dataloader(
        img_dir=path,
        # (view_1, view_2) tuple
        num_views=2, 
        shuffle=True,
        transforms=[
            # Scaling to a minimum of 0.75
            v2.RandomResizedCrop(scale=(0.75, 1.0), size=cfg["img_size"]),
            # Translation 5% (0.05), Rotation max 15 degrees
            v2.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05)),
            # Horizontal Flip
            v2.RandomHorizontalFlip(p=0.5)
        ]
    )
