"""
Prediction pipeline combining classifier and regressor.

"""
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List, Any

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.transforms import get_inference_transforms
from models.classifier import FoodClassifier
from models.regressor import CalorieRegressor


class FoodCaloriePipeline:
    """
    End-to-end pipeline for food classification and calorie estimation.
    
    """
    
    def __init__(
        self,
        classifier_path: Optional[str] = None,
        regressor_path: Optional[str] = None,
        device: Optional[str] = None,
        calorie_scale: float = 1000.0,
    ):
        """
        Args:
            classifier_path: Path to trained classifier checkpoint
            regressor_path: Path to trained regressor checkpoint
            device: Device to run inference on
            calorie_scale: Scale used during regressor training
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.calorie_scale = calorie_scale
        
        # Load models
        self.classifier = None
        self.regressor = None
        self.class_names = None
        
        if classifier_path:
            print("starting to load classifier")
            self._load_classifier(classifier_path)
        
        if regressor_path:
            print("starting to load regressor")
            self._load_regressor(regressor_path)
        
        #Standardize input by resizing ot 224 x 224, then normalize with ImageNet mean and std
        self.transform = get_inference_transforms()
    
    def _load_classifier(self, path: str):
        """Load trained classifier."""
        print("in classifier loading method")
        #get best model
        checkpoint = torch.load(path, map_location=self.device, weights_only = False) 
        config = checkpoint.get("config", {})
        
        #intialize model and load trained parameters
        self.classifier = FoodClassifier(
            num_classes=config.get("num_classes", 101),
            backbone=config.get("backbone", "resnet50"),
            dropout=config.get("dropout", 0.3),
        )
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        #classes that food can be sorted into
        self.class_names = [
            'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
            'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
            'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
            'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla',
            'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
            'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
            'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
            'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
            'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
            'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
            'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
            'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
            'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
            'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
            'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
            'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
            'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
            'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
            'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
            'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
        ]
    

        print(f"Loaded classifier from {path}")
    
    def _load_regressor(self, path: str):
        """Load trained regressor."""

        #get best regresor params
        checkpoint = torch.load(path, map_location=self.device, weights_only = False)
        config = checkpoint.get("config", {})
        print("attempting to load regressor")
        
        #intialize regressor and load trained weights
        self.regressor = CalorieRegressor(
            backbone=config.get("backbone", "resnet50"),
            freeze_backbone=True,
            hidden_dim=config.get("hidden_dim", 256),
            dropout=config.get("dropout", 0.2),
        )
        self.regressor.load_state_dict(checkpoint["model_state_dict"])
        self.regressor = self.regressor.to(self.device)
        self.regressor.eval()
        
        print(f"Loaded regressor from {path}")
    
    def _preprocess(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Load and preprocess image."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)
    
    @torch.no_grad()
    def classify(self, image: Union[str, Path, Image.Image]) -> Dict:
        """
        Classify food image.
        
        Returns:
            Dict with class_name, class_idx, confidence, top5_classes
        """
        if self.classifier is None:
            raise ValueError("Classifier not loaded")
        
        #pre-process image, compute logits and probability for classification
        tensor = self._preprocess(image)
        logits = self.classifier(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        
        # Top-1
        top1_prob, top1_idx = probs.max(0)
        
        # Top-5
        top5_probs, top5_indices = probs.topk(5)
        
        # Map indices to class names
        if self.class_names:
            top1_name = self.class_names[top1_idx.item()]
            top5_classes = [
                {"class_name": self.class_names[idx.item()], "confidence": prob.item()}
                for idx, prob in zip(top5_indices, top5_probs)
            ]
        else:
            top1_name = f"class_{top1_idx.item()}"
            top5_classes = [
                {"class_name": f"class_{idx.item()}", "confidence": prob.item()}
                for idx, prob in zip(top5_indices, top5_probs)
            ]
        
        return {
            "class_name": top1_name,
            "class_idx": top1_idx.item(),
            "confidence": top1_prob.item(),
            "top5_classes": top5_classes,
        }
    
    @torch.no_grad()
    def estimate_calories_regression(self, image: Union[str, Path, Image.Image]) -> float:
        """
        Estimate calories using regression model.
        
        Returns:
            Predicted calories in kcal
        """
        if self.regressor is None:
            raise ValueError("Regressor not loaded")
        
        #preprocess image, predict calorie count
        tensor = self._preprocess(image)
        output = self.regressor(tensor)
        calories = output.item() * self.calorie_scale
        
        return calories
    
    def predict(
        self,
        image: Union[str, Path, Image.Image],
    ) -> Dict:
        """
        Full prediction: classify + estimate calories.
        
        Args:
            image: Input food image

        Returns:
            Dict with class_name, confidence, calories
        """
        result: Dict[str, Any]={
            "class_name": None,
            "confidence": None,
            "calories": None,
        }
        
        # Classification
        if self.classifier:
            class_result = self.classify(image)
            result["class_name"] = class_result["class_name"]
            result["confidence"] = class_result["confidence"]
            result["top5_classes"] = class_result["top5_classes"]
        else:
            print("no classifier")
        
        if self.regressor:
            result["calories"] = self.estimate_calories_regression(image)
        else:
            print("no regressor")
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        **kwargs
    ) -> List[Dict]:
        """Predict on multiple images."""
        return [self.predict(img, **kwargs) for img in images]


def demo():
    """Demo usage of the pipeline."""
    print("FoodCaloriePipeline Demo")
    print("=" * 50)
    
    #pipline initialization
    pipeline = FoodCaloriePipeline(
        classifier_path="artifacts/models/classifier_best.pth",
        regressor_path="artifacts/models/regressor_best.pth",
    )
    
    strategy = input("Enter")
    # Predict on a single image
    result = pipeline.predict("path/to/food/image.jpg")
    
    print(f"Predicted class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Estimated calories: {result['calories']:.0f} kcal")


if __name__ == "__main__":
    demo()
