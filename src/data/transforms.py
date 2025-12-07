"""
Data transforms for Food-101 classification and calorie regression.
"""
import torchvision.transforms as T

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(img_size: int = 224) -> T.Compose:
    """
    Training transforms with augmentation.
    """
    return T.Compose([
        T.Resize((img_size + 32, img_size + 32)),  # Resize slightly larger
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transforms(img_size: int = 224) -> T.Compose:
    """
    Validation/test transforms (no augmentation).
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_inference_transforms(img_size: int = 224) -> T.Compose:
    """
    Inference transforms for single images.
    """
    return get_val_transforms(img_size)

def denormalize(tensor):
    """
    Reverse normalization for visualization.
    """
    mean = T.functional.to_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = T.functional.to_tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean
