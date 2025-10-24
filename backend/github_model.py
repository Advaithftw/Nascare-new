"""
GitHub Model Architecture - EfficientNet-B4
Based on: enrico310786/brain_tumor_classification
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights

def create_github_model(num_classes=4, n_nodes=256, dropout=0.5):
    """
    Creates the EfficientNet-B4 model architecture from the GitHub repository.
    Based on: enrico310786/brain_tumor_classification/image_classification_model.py
    
    Architecture:
    - EfficientNet-B4 base model (1792 feature channels)
    - AdaptiveAvgPool2d(1, 1)
    - Custom classifier head:
      * Flatten
      * Linear(1792 → n_nodes)
      * ReLU
      * Dropout(dropout)
      * Linear(n_nodes → num_classes)
    
    Args:
        num_classes: Number of output classes (default: 4 for brain tumor types)
        n_nodes: Hidden layer size in classifier (default: 512)
        dropout: Dropout probability (default: 0.5)
    
    Returns:
        Configured EfficientNet-B4 model
    """
    # Create EfficientNet-B4 base model without pretrained weights
    # (we'll load from checkpoint)
    base_model = models.efficientnet_b4(weights=None)
    
    # Replace avgpool
    base_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    # Create custom classifier head (matches GitHub repository)
    # EfficientNet-B4 features output: 1792 channels
    base_model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1792, n_nodes),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(n_nodes, num_classes)
    )
    
    return base_model

def load_github_checkpoint(checkpoint_path, num_classes=4, n_nodes=256, dropout=0.5, device='cpu'):
    """
    Load the best.pth checkpoint with EfficientNet-B4 architecture.
    
    Checkpoint structure (from GitHub repo):
    {
        'model': state_dict,
        'optimizer': optimizer_state,
        'scheduler': scheduler_state,
        'epoch': int,
        'best_eva_accuracy': float
    }
    
    Args:
        checkpoint_path: Path to best.pth file
        num_classes: Number of output classes
        n_nodes: Hidden layer size (must match training config)
        dropout: Dropout probability (must match training config)
        device: Device to load model on
    
    Returns:
        Loaded model with trained weights
    """
    # Create the model architecture
    model = create_github_model(num_classes=num_classes, n_nodes=n_nodes, dropout=dropout)
    model = model.to(device)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state_dict (it's stored under 'model' key)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        
        # Strip 'model.' prefix from keys (checkpoint has model.features.*, model.classifier.*)
        # but our model expects features.*, classifier.*
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Load the state_dict
        model.load_state_dict(new_state_dict)
        
        # Set to eval mode by default
        # Note: If you get poor results in eval mode, you may need to use train mode
        # for BatchNorm layers due to running statistics issues
        model.eval()
        
        print(f"✓ Loaded model checkpoint")
    else:
        raise ValueError("Unexpected checkpoint format - missing 'model' key")
    
    return model

# Example usage:
if __name__ == "__main__":
    import os
    
    # Test loading the model
    checkpoint_path = "best.pth"
    
    if os.path.exists(checkpoint_path):
        print("Loading GitHub model (EfficientNet-B4)...")
        model = load_github_checkpoint(checkpoint_path)
        print("Model loaded successfully!")
        print(f"\nModel structure:\n{model}")
    else:
        print(f"Checkpoint not found at: {checkpoint_path}")
        print("\nYou can still create the architecture:")
        model = create_github_model()
        print("Empty model created (needs trained weights)")

