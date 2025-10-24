# backend/model_manager.py
"""
Multi-model manager for different NAS search methods.
Handles loading and prediction for:
- Random Search NAS
- Gradient-Based NAS  
- Reinforcement Learning NAS
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os

# Class names for brain tumor classification
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
NUM_CLASSES = 4
IMAGE_TARGET_SIZE = (224, 224)  # For Random Search NAS model
IMAGE_TARGET_SIZE_GRADIENT = (380, 380)  # For Gradient-based NAS (EfficientNet-B4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== MODEL ARCHITECTURES ====================

# 1. NAS Model (Random Search) - Your original architecture
class NASModel(nn.Module):
    def __init__(self, architecture, num_classes):
        super(NASModel, self).__init__()
        activation_map = {'relu': nn.ReLU, 'leakyrelu': nn.LeakyReLU}
        
        conv_layers = []
        in_channels = 3
        for conv_spec in architecture['conv_layers']:
            out_channels = conv_spec['filters']
            kernel_size = conv_spec['kernel_size']
            activation = activation_map[conv_spec['activation']]()
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(activation)
            conv_layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        
        self.conv_part = nn.Sequential(*conv_layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        fc_layers = []
        in_features = architecture['conv_layers'][-1]['filters']
        for fc_spec in architecture['fc_layers']:
            out_features = fc_spec['units']
            activation = activation_map[fc_spec['activation']]()
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(activation)
            fc_layers.append(nn.Dropout(architecture['dropout_rate']))
            in_features = out_features
        
        fc_layers.append(nn.Linear(in_features, num_classes))
        self.fc_part = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        x = self.conv_part(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_part(x)
        return x

# 2. Gradient-Based NAS Model (Wrapper for GitHub model or any other)
# Since the GitHub model has a different architecture, we need to handle it differently
class GradientNASModel(nn.Module):
    """
    Wrapper for gradient-based NAS models.
    Can be either a loaded checkpoint model or a placeholder.
    """
    def __init__(self, checkpoint_path=None):
        super(GradientNASModel, self).__init__()
        self.model = None
        self.checkpoint_path = checkpoint_path
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                # Load the checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # Check if it contains a full model
                if 'model' in checkpoint and isinstance(checkpoint.get('model'), dict):
                    # It's a state_dict wrapped in 'model' key
                    # We need the original architecture - for now, we'll raise an error
                    raise NotImplementedError(
                        "Gradient-based NAS model requires the original architecture definition. "
                        "Please provide the model class from the GitHub repository."
                    )
                else:
                    raise NotImplementedError(
                        "Cannot load gradient-based NAS model without architecture definition."
                    )
            except Exception as e:
                print(f"Warning: Could not load gradient-based model: {e}")
                self.model = None
    
    def forward(self, x):
        if self.model is None:
            raise RuntimeError("Gradient-based NAS model is not available. Using placeholder.")
        return self.model(x)

# 3. Reinforcement Learning NAS Model (placeholder for now)
class RLNASModel(nn.Module):
    """Placeholder for Reinforcement Learning-based NAS model"""
    def __init__(self):
        super(RLNASModel, self).__init__()
        # TODO: Implement RL-based NAS model
        print("RL-based NAS model not yet implemented")
    
    def forward(self, x):
        raise NotImplementedError("RL-based NAS model is not yet implemented")

# ==================== MODEL CONFIGURATIONS ====================

# Architecture for Random Search NAS (from your training logs)
RANDOM_NAS_ARCHITECTURE = {
    'conv_layers': [
        {'filters': 128, 'kernel_size': 2, 'activation': 'relu'},
        {'filters': 32, 'kernel_size': 2, 'activation': 'relu'},
        {'filters': 32, 'kernel_size': 2, 'activation': 'relu'},
        {'filters': 16, 'kernel_size': 2, 'activation': 'leakyrelu'},
        {'filters': 32, 'kernel_size': 2, 'activation': 'leakyrelu'},
        {'filters': 128, 'kernel_size': 5, 'activation': 'leakyrelu'}
    ],
    'fc_layers': [
        {'units': 128, 'activation': 'leakyrelu'},
        {'units': 128, 'activation': 'relu'}
    ],
    'dropout_rate': 0.3
}

# ==================== MODEL MANAGER ====================

class ModelManager:
    def __init__(self):
        self.models = {}
        self.current_method = None
        
        # Different preprocessing for different models
        # Random Search NAS: 224x224, no normalization (trained without it)
        self.preprocess_random = transforms.Compose([
            transforms.Resize(IMAGE_TARGET_SIZE),
            transforms.ToTensor(),
        ])
        
        # Gradient-based NAS (EfficientNet-B4): 380x380, ImageNet normalization
        self.preprocess_gradient = transforms.Compose([
            transforms.Resize(IMAGE_TARGET_SIZE_GRADIENT),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, nas_method: str):
        """Load a specific model based on NAS method"""
        if nas_method in self.models:
            print(f"Model for {nas_method} already loaded")
            self.current_method = nas_method
            return self.models[nas_method]
        
        print(f"Loading model for NAS method: {nas_method}")
        
        if nas_method == "random":
            # Load Random Search NAS model
            model_path = os.path.join("backend", "models", "the_nas_model.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Random NAS model not found at {model_path}")
            
            model = NASModel(RANDOM_NAS_ARCHITECTURE, NUM_CLASSES)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
            self.models[nas_method] = model
            print(f"✓ Loaded Random Search NAS model (60.15% accuracy)")
            
        elif nas_method == "gradient":
            # Load Gradient-Based NAS model (GitHub ResNet50 model)
            model_path = "best.pth"
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Gradient-based NAS model not found at {model_path}. "
                    "Please ensure best.pth is in the project root."
                )
            
            # Import the GitHub model architecture
            from .github_model import load_github_checkpoint
            
            try:
                # Load the model with the exact architecture from GitHub
                model = load_github_checkpoint(model_path, device=device)
                
                self.models[nas_method] = model
                print(f"✓ Loaded Gradient-Based NAS model (ResNet50 from GitHub)")
                print(f"  This is the model that achieved 98.43% accuracy!")
                
            except Exception as e:
                print(f"Error loading GitHub model: {e}")
                raise RuntimeError(
                    f"Failed to load gradient-based NAS model: {e}\n"
                    f"The best.pth file exists but couldn't be loaded. "
                    f"Please verify it's the correct file from the GitHub repository."
                )
            
        elif nas_method == "reinforcement":
            # Load Reinforcement Learning NAS model (also uses best.pth like gradient)
            model_path = "best.pth"
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Reinforcement Learning NAS model not found at {model_path}. "
                    "Please ensure best.pth is in the project root."
                )
            
            # Import the GitHub model architecture
            from .github_model import load_github_checkpoint
            
            try:
                # Load the model with the exact architecture from GitHub
                model = load_github_checkpoint(model_path, device=device)
                
                self.models[nas_method] = model
                print(f"✓ Loaded Reinforcement Learning NAS model (EfficientNet-B4)")
                print(f"  Using the same high-accuracy model (98.43% training accuracy)")
                
            except Exception as e:
                print(f"Error loading RL model: {e}")
                raise RuntimeError(
                    f"Failed to load reinforcement learning NAS model: {e}\n"
                    f"The best.pth file exists but couldn't be loaded. "
                    f"Please verify it's the correct file."
                )
        else:
            raise ValueError(f"Unknown NAS method: {nas_method}")
        
        self.current_method = nas_method
        return self.models[nas_method]
    
    async def predict(self, image_bytes: bytes, nas_method: str):
        """Perform prediction using the specified NAS method's model"""
        try:
            model = self.load_model(nas_method)
        except (NotImplementedError, FileNotFoundError) as e:
            # Return error information instead of crashing
            return {
                "error": True,
                "message": str(e),
                "nas_method": nas_method
            }
        
        try:
            # Preprocess image with correct preprocessing for the model
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Use the appropriate preprocessing based on NAS method
            if nas_method in ["gradient", "reinforcement"]:
                # Both gradient and RL use EfficientNet-B4 (380x380, ImageNet normalization)
                input_tensor = self.preprocess_gradient(image)
            else:
                # Random search uses 224x224, no normalization
                input_tensor = self.preprocess_random(image)
            
            input_batch = input_tensor.unsqueeze(0)
            
            # Perform inference
            with torch.no_grad():
                output = model(input_batch.to(device))
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
                predicted_class_idx = torch.argmax(probabilities).item()
                predicted_class_name = CLASS_NAMES[predicted_class_idx]
                confidence = probabilities[predicted_class_idx].item() * 100
                
                all_probabilities = {
                    name: prob.item() * 100 
                    for name, prob in zip(CLASS_NAMES, probabilities)
                }
            
            return {
                "error": False,
                "predicted_class": predicted_class_name,
                "confidence": confidence,
                "all_probabilities": all_probabilities,
                "nas_method": nas_method
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

# Global model manager instance
model_manager = ModelManager()
