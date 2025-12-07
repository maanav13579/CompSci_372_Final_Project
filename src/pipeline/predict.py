"""
Unified prediction pipeline combining classifier and regressor.

Usage:
    from pipeline.predict import FoodCaloriePipeline
    
    pipeline = FoodCaloriePipeline(
        classifier_path="artifacts/models/classifier_best.pth",
        regressor_path="artifacts/models/regressor_best.pth",
        calorie_map_path="data/calorie_map.csv"
    )
    
    result = pipeline.predict("path/to/food/image.jpg")
"""
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

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
    
    Supports multiple integration strategies:
    1. Classifier only + calorie lookup
    2. Regressor only
    3. Ensemble (weighted combination)
    4. Confidence-based switching
    """
    
    def __init__(
        self,
        classifier_path: Optional[str] = None,
        regressor_path: Optional[str] = None,
        calorie_map_path: Optional[str] = None,
        device: Optional[str] = None,
        calorie_scale: float = 1000.0,
    ):
        """
        Args:
            classifier_path: Path to trained classifier checkpoint
            regressor_path: Path to trained regressor checkpoint
            calorie_map_path: Path to CSV with class -> calorie mapping
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
        self.calorie_map = None
        self.class_names = None
        
        if classifier_path:
            self._load_classifier(classifier_path)
        
        if regressor_path:
            self._load_regressor(regressor_path)
        
        if calorie_map_path:
            self._load_calorie_map(calorie_map_path)
        
        # Transforms
        self.transform = get_inference_transforms()
    
    def _load_classifier(self, path: str):
        """Load trained classifier."""
        checkpoint = torch.load(path, map_location=self.device)
        config = checkpoint.get("config", {})
        
        self.classifier = FoodClassifier(
            num_classes=config.get("num_classes", 101),
            backbone=config.get("backbone", "resnet50"),
            dropout=config.get("dropout", 0.3),
        )
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        print(f"Loaded classifier from {path}")
    
    def _load_regressor(self, path: str):
        """Load trained regressor."""
        checkpoint = torch.load(path, map_location=self.device)
        config = checkpoint.get("config", {})
        
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
    
    def _load_calorie_map(self, path: str):
        """Load calorie lookup table."""
        df = pd.read_csv(path)
        self.calorie_map = dict(zip(df["class_name"], df["calories_per_serving"]))
        self.class_names = list(df["class_name"])
        print(f"Loaded calorie map with {len(self.calorie_map)} classes")
    
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
        
        tensor = self._preprocess(image)
        output = self.regressor(tensor)
        calories = output.item() * self.calorie_scale
        
        return calories
    
    def estimate_calories_lookup(self, class_name: str) -> Optional[float]:
        """
        Get calorie estimate from lookup table.
        
        Returns:
            Calories from lookup table, or None if not found
        """
        if self.calorie_map is None:
            return None
        
        return self.calorie_map.get(class_name)
    
    def predict(
        self,
        image: Union[str, Path, Image.Image],
        strategy: str = "ensemble",
        confidence_threshold: float = 0.7,
        ensemble_weight: float = 0.5,
    ) -> Dict:
        """
        Full prediction: classify + estimate calories.
        
        Args:
            image: Input food image
            strategy: One of:
                - 'classifier_only': Use classifier + lookup table
                - 'regressor_only': Use regressor only
                - 'ensemble': Weighted combination of lookup and regression
                - 'confidence_switch': Use lookup if classifier confident, else regressor
            confidence_threshold: Threshold for confidence-based switching
            ensemble_weight: Weight for regressor in ensemble (1-weight for lookup)
        
        Returns:
            Dict with class_name, confidence, calories, method
        """
        result = {
            "class_name": None,
            "confidence": None,
            "calories": None,
            "method": strategy,
        }
        
        # Classification
        if self.classifier:
            class_result = self.classify(image)
            result["class_name"] = class_result["class_name"]
            result["confidence"] = class_result["confidence"]
            result["top5_classes"] = class_result["top5_classes"]
        
        # Calorie estimation based on strategy
        if strategy == "classifier_only":
            if result["class_name"] and self.calorie_map:
                result["calories"] = self.estimate_calories_lookup(result["class_name"])
                result["calorie_method"] = "lookup"
        
        elif strategy == "regressor_only":
            if self.regressor:
                result["calories"] = self.estimate_calories_regression(image)
                result["calorie_method"] = "regression"
        
        elif strategy == "ensemble":
            lookup_cal = None
            regression_cal = None
            
            if result["class_name"] and self.calorie_map:
                lookup_cal = self.estimate_calories_lookup(result["class_name"])
            
            if self.regressor:
                regression_cal = self.estimate_calories_regression(image)
            
            if lookup_cal is not None and regression_cal is not None:
                result["calories"] = (
                    (1 - ensemble_weight) * lookup_cal + ensemble_weight * regression_cal
                )
                result["calorie_method"] = "ensemble"
                result["lookup_calories"] = lookup_cal
                result["regression_calories"] = regression_cal
            elif lookup_cal is not None:
                result["calories"] = lookup_cal
                result["calorie_method"] = "lookup"
            elif regression_cal is not None:
                result["calories"] = regression_cal
                result["calorie_method"] = "regression"
        
        elif strategy == "confidence_switch":
            use_lookup = (
                result["confidence"] is not None
                and result["confidence"] >= confidence_threshold
                and result["class_name"] is not None
                and self.calorie_map is not None
            )
            
            if use_lookup:
                result["calories"] = self.estimate_calories_lookup(result["class_name"])
                result["calorie_method"] = "lookup (high confidence)"
            elif self.regressor:
                result["calories"] = self.estimate_calories_regression(image)
                result["calorie_method"] = "regression (low confidence)"
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        strategy: str = "ensemble",
        **kwargs
    ) -> List[Dict]:
        """Predict on multiple images."""
        return [self.predict(img, strategy=strategy, **kwargs) for img in images]


def demo():
    """Demo usage of the pipeline."""
    print("FoodCaloriePipeline Demo")
    print("=" * 50)
    
    # Example usage (with placeholder paths)
    pipeline = FoodCaloriePipeline(
        classifier_path="artifacts/models/classifier_best.pth",
        regressor_path="artifacts/models/regressor_best.pth",
        calorie_map_path="data/calorie_map.csv",
    )
    
    # Predict on a single image
    result = pipeline.predict("path/to/food/image.jpg", strategy="ensemble")
    
    print(f"Predicted class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Estimated calories: {result['calories']:.0f} kcal")
    print(f"Method: {result['calorie_method']}")


if __name__ == "__main__":
    demo()
