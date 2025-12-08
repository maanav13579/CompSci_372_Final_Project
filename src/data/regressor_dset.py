"""
Wrapper for calorie regression dataset (Nutrition5k).
"""
import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np


class CalorieDataset(Dataset):
    """
    Dataset for calorie regression.
    """
    
    def __init__(
        self,
        image_paths: List[Path],
        calories: List[float],
        transform: Optional[Callable] = None,
        calorie_scale: float = 1000.0,  # Normalize calories
    ):
        """
        Args:
            image_paths: List of paths to images
            calories: List of calorie values (in kcal)
            transform: Optional transform to apply to images
            calorie_scale: Scale factor for normalizing calories
        """
        self.image_paths = image_paths
        self.calories = calories
        self.transform = transform
        self.calorie_scale = calorie_scale
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        calorie = self.calories[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Normalize calorie value
        calorie_normalized = torch.tensor(calorie / self.calorie_scale, dtype=torch.float32)
        
        return image, calorie_normalized


class Nutrition5kDataset(Dataset):
    """
    Nutrition5k dataset loader.
    
    The below diagram was generated using AI-Assistant, prompted with a text description of the file structure
    Expected structure:
        nutrition5k/
        ├── imagery/
        │   └── realsense_overhead/
        │       ├── dish_1551290117/
        │       │   ├── rgb.png          <- Only this is used. Overhead color picture. 
        │       │   ├── depth_color.png  <- Ignored
        │       │   └── depth_raw.png    <- Ignored
        │       └── ...
        ├── metadata/
        │   └── dish_metadata_cafe1.csv
        └── splits/
            ├── rgb_train_ids.txt
            └── rgb_test_ids.txt
            
    The metadata CSV contains columns like:
        dish_id, total_calories, total_mass, total_fat, total_carb, total_protein
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        calorie_scale: float = 1000.0,
        val_ratio: float = 0.1,
    ):
        """
        Args:
            root: Path to nutrition5k folder
            split: One of 'train', 'val', 'test'
            transform: Optional image transforms
            calorie_scale: Scale factor for normalizing calories
            val_ratio: Fraction of training data to use for validation
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.calorie_scale = calorie_scale
        
        # Load samples using official splits
        self.samples = self._load_samples(val_ratio)
        
    def _load_samples(self, val_ratio: float) -> List[Tuple[Path, float]]:
        """Load image paths and calorie labels using official train/test splits."""
        
        # Find split files (provided by nutrition5k developers)
        train_ids_file = self.root / "splits" / "rgb_train_ids.txt"
        test_ids_file = self.root / "splits" / "rgb_test_ids.txt"
        
        if not train_ids_file.exists():
            raise FileNotFoundError(f"Could not find {train_ids_file}")
        if not test_ids_file.exists():
            raise FileNotFoundError(f"Could not find {test_ids_file}")
        
        # Load dish IDs for each split
        with open(train_ids_file, 'r') as f:
            train_ids = set(line.strip() for line in f if line.strip())
        with open(test_ids_file, 'r') as f:
            test_ids = set(line.strip() for line in f if line.strip())
        
        print(f"Loaded {len(train_ids)} train IDs and {len(test_ids)} test IDs from split files")
        
        # Find metadata files provided by nutrition5k developers
        metadata_candidates = [
            self.root / "metadata" / "dish_metadata_cafe1.csv",
            self.root / "metadata" / "dish_metadata_cafe2.csv",
        ]
        
        metadata_file = None
        for candidate in metadata_candidates:
            if candidate.exists():
                metadata_file = candidate
                break
        
        if metadata_file is None:
            raise FileNotFoundError(
                f"Could not find Nutrition5k metadata CSV in {self.root}. "
                f"Checked: {[str(c) for c in metadata_candidates]}"
            )
        
        print(f"Loading metadata from: {metadata_file}")
        
        # Load CSV - variable number of columns per row (ingredients vary)
        # Only need first two columns: dish_id, total_calories
        dish_ids = []
        calories = []
        
        #Extract first two columns from metadata csv
        with open(metadata_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')

                #extract only valid columns with food_id and calorie information
                if len(parts) >= 2:
                    dish_ids.append(parts[0])
                    try:
                        cal = float(parts[1])
                        calories.append(cal)
                    except ValueError:
                        calories.append(None)
                else:
                    dish_ids.append(None)
                    calories.append(None)
        
        df = pd.DataFrame({'dish_id': dish_ids, 'total_calories': calories})
        
        dish_id_col = 'dish_id'
        calorie_col = 'total_calories'
        
        print(f"Loaded {len(df)} rows from metadata")
        print(f"Using columns: dish_id='{dish_id_col}', calories='{calorie_col}'")
        
        # Create dish_id calorie mapping
        calorie_map = {}
        for _, row in df.iterrows():
            dish_id = str(row[dish_id_col])
            calories = row[calorie_col]
            if not pd.isna(calories) and calories > 0:
                calorie_map[dish_id] = float(calories)
        
        print(f"Loaded {len(calorie_map)} dishes with valid calorie data")
        

        imagery_root = self.root / "imagery" / "realsense_overhead"
        
        if not imagery_root.exists():
            raise FileNotFoundError(f"Could not find imagery directory in {self.root}")
        
        print(f"Loading images from: {imagery_root}")
        
        # Build samples list based on split
        if self.split == "test":
            dish_ids_to_use = test_ids
        else:
            dish_ids_to_use = train_ids
        
        samples = []
        skipped = 0
        
        for dish_id in dish_ids_to_use:
            # Check if we have calorie data
            if dish_id not in calorie_map:
                skipped += 1
                continue
            
            # Find rgb.png image (ignore depth images)
            img_candidates = [
                imagery_root / dish_id / "rgb.png",
                imagery_root / f"dish_{dish_id}" / "rgb.png",
            ]
            
            img_found = False
            for img_path in img_candidates:
                if img_path.exists():
                    samples.append((img_path, calorie_map[dish_id]))
                    img_found = True
                    break
            
            if not img_found:
                skipped += 1
        
        print(f"Found {len(samples)} valid samples, skipped {skipped}")
                
        # For train split, further divide into train/val
        if self.split in ["train", "val"]:
            np.random.seed(42)
            indices = np.random.permutation(len(samples))
            n_val = int(len(samples) * val_ratio)
            
            if self.split == "val":
                indices = indices[:n_val]
            else:  
                indices = indices[n_val:]
            
            samples = [samples[i] for i in indices]
        
        print(f"Split '{self.split}': {len(samples)} samples")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns transformed and normalized image calorie pair from list of samples
        '''

        img_path, calorie = self.samples[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        calorie_normalized = torch.tensor(calorie / self.calorie_scale, dtype=torch.float32)
        
        return image, calorie_normalized


def get_calorie_dataloaders(
    dataset_type: str,
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for calorie regression.
    
    Args:
        dataset_type: 'nutrition5k'
        root: Path to dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from .transforms import get_train_transforms, get_val_transforms
    
    if train_transform is None:
        train_transform = get_train_transforms()
    if val_transform is None:
        val_transform = get_val_transforms()
    
    #create train, val, test splits using the above classes
    if dataset_type == "nutrition5k":
        train_dataset = Nutrition5kDataset(root, split="train", transform=train_transform)
        val_dataset = Nutrition5kDataset(root, split="val", transform=val_transform)
        test_dataset = Nutrition5kDataset(root, split="test", transform=val_transform)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    #create the dataloaders for each split
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
