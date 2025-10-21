# backend/model.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os

# --- Your NASModel definition (copied directly from your newtraining.py) ---
class NASModel(nn.Module):
    def __init__(self, architecture, num_classes):
        super(NASModel, self).__init__()
        conv_layers = []
        in_channels = 3  # RGB input (assuming MRI images are converted to 3 channels)
        for conv_spec in architecture['conv_layers']:
            out_channels = conv_spec['filters']
            kernel_size = conv_spec['kernel_size']
            # Map activation string to PyTorch nn.Module
            activation = nn.ReLU() if conv_spec['activation'] == 'relu' else nn.LeakyReLU()
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(activation)
            conv_layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        
        # The convolutional part of the network
        self.conv_part = nn.Sequential(*conv_layers)
        
        # Adaptive average pooling to get a fixed-size output regardless of input image size
        # This pools each feature map to a single value, making the output (Batch_size, last_conv_filters, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1) 
        
        fc_layers = []
        # Input features for the first FC layer come from the number of filters of the last convolutional layer
        # after the global average pooling.
        in_features_for_first_fc = architecture['conv_layers'][-1]['filters'] 

        # Iterate through the defined fully connected layers in the architecture
        for i, fc_spec in enumerate(architecture['fc_layers']):
            out_features = fc_spec['units']
            activation = nn.ReLU() if fc_spec['activation'] == 'relu' else nn.LeakyReLU()
            
            # Determine input features for the current FC layer:
            # If it's the first FC layer (i=0), its input comes from the last conv layer's output.
            # Otherwise, its input comes from the output of the *previous* FC layer.
            current_in_features = in_features_for_first_fc if i == 0 else architecture['fc_layers'][i-1]['units']

            fc_layers.append(nn.Linear(current_in_features, out_features))
            fc_layers.append(activation)
            fc_layers.append(nn.Dropout(architecture['dropout_rate']))
        
        # The final classification layer maps from the output of the *last* FC layer in the list to num_classes
        final_in_features = architecture['fc_layers'][-1]['units']
        fc_layers.append(nn.Linear(final_in_features, num_classes))
        
        # The fully connected part of the network
        self.fc_part = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        # Pass input through convolutional layers
        x = self.conv_part(x)
        # Apply global average pooling
        x = self.global_pool(x)
        # Flatten the output for the fully connected layers (remove 1x1 spatial dimensions)
        x = x.view(x.size(0), -1) # Reshapes to (Batch_size, num_features)
        # Pass through fully connected layers
        x = self.fc_part(x)
        return x
# --- End of NASModel definition ---

# Path to your trained model weights file
# Prefer a top-level `best.pth` if present (convenient), otherwise fall back to backend/models/the_nas_model.pth
ROOT_BEST = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "best.pth"))
FALLBACK_MODEL = os.path.join(os.path.dirname(__file__), "models", "the_nas_model.pth")
if os.path.exists(ROOT_BEST):
    MODEL_PATH = ROOT_BEST
else:
    MODEL_PATH = FALLBACK_MODEL

# Number of tumor classes (4 based on your project description)
NUM_CLASSES = 4 

# Class names in the order they were indexed (0, 1, 2, 3) during training
# These names are crucial for mapping model output to human-readable labels
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# The specific architecture found by your NAS and saved in the log
# This dictionary precisely defines the structure of the model saved in best_final_model.pt
# It MUST EXACTLY MATCH the 'Best architecture found:' entry from your nas_training.log
BEST_MODEL_ARCHITECTURE = {
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

# Global variable to hold the loaded model instance
model_instance = None 
# Determine the device for PyTorch inference (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """
    Loads the trained PyTorch model into memory. This function is designed to be called
    once at application startup to avoid reloading the model for every request.
    """
    global model_instance
    if model_instance is None:
        print(f"Loading model from {MODEL_PATH} on device: {device}...")
        
        # Instantiate the NASModel with the correct architecture retrieved from the logs
        model_instance = NASModel(BEST_MODEL_ARCHITECTURE, num_classes=NUM_CLASSES)
        
        # Check if the model weights file actually exists at the specified path
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model weights not found at: {MODEL_PATH}. "
                                    f"Place your .pth file at the project root as 'best.pth' or in 'backend/models/'.")

        # Load the file. It may be:
        # - a state_dict (mapping of parameter names -> tensors)
        # - a checkpoint dict like {'model': state_dict, 'optimizer': ...}
        # - a pickled full model (nn.Module)
        loaded = torch.load(MODEL_PATH, map_location=device)
        if isinstance(loaded, nn.Module):
            # Pickled full model object
            print("Loaded a full model object from disk. Using it directly (this requires compatible class definitions).")
            model_instance = loaded
        elif isinstance(loaded, dict):
            # If it's a checkpoint wrapper, try common keys
            nested = None
            for key in ("model", "state_dict", "model_state_dict", "state"):
                if key in loaded and isinstance(loaded[key], dict):
                    nested = loaded[key]
                    break
            state_dict = nested if nested is not None else loaded
            # Attempt to load state_dict into our NASModel
            # First try a strict load to get a clear error if keys match exactly
            try:
                model_instance.load_state_dict(state_dict)
            except Exception as e_strict:
                # If that fails, attempt some smart fallbacks:
                # 1) If the checkpoint contained a full model under 'model', use it
                if isinstance(loaded.get("model"), nn.Module):
                    print("Checkpoint contains a full model under key 'model'. Using that model object.")
                    model_instance = loaded.get("model")
                else:
                    # 2) Try stripping common prefixes like 'model.' or 'module.' from keys and load with strict=False
                    def strip_prefix_from_keys(sd, prefixes=("model.", "module.")):
                        new = {}
                        for k, v in sd.items():
                            new_k = k
                            for p in prefixes:
                                if k.startswith(p):
                                    new_k = k[len(p):]
                                    break
                            new[new_k] = v
                        return new

                    stripped = strip_prefix_from_keys(state_dict)
                    try:
                        res = model_instance.load_state_dict(stripped, strict=False)
                        # res is a namedtuple with missing_keys and unexpected_keys
                        print("Loaded with stripped prefixes (strict=False).\nMissing keys:", getattr(res, 'missing_keys', None), "\nUnexpected keys:", getattr(res, 'unexpected_keys', None))
                    except Exception:
                        # 3) Report helpful diagnostic and re-raise the original strict error
                        top_keys = list(loaded.keys())
                        raise RuntimeError(
                            f"Failed to load state_dict into NASModel (strict error: {e_strict}).\n"
                            f"Tried stripping common prefixes and strict=False but it still failed.\n"
                            f"Top-level keys in the checkpoint: {top_keys}\n"
                            f"If this is a checkpoint from a different architecture you will need to either: \n"
                            f"  - load and use the original model object saved in the checkpoint, or\n"
                            f"  - re-save a state_dict from the model matching NASModel architecture, or\n"
                            f"  - provide a mapping between parameter names."
                        )
        else:
            # Unexpected object
            raise RuntimeError(f"Unrecognized model file format for {MODEL_PATH}: {type(loaded)}")
        model_instance.to(device) # Move the model to the determined device (CPU/GPU)
        model_instance.eval() # Set model to evaluation mode (disables dropout, batchnorm updates)
        print("Model loaded successfully and set to evaluation mode.")
    return model_instance

# Define image preprocessing transformations
# Based on your newtraining.py's load_data, images were:
# 1. Opened as RGB (Image.open(...).convert('RGB'))
# 2. Converted to numpy array and scaled by /255.0
# 3. Transposed to (Channels, Height, Width)
# transforms.ToTensor() handles steps 2 and 3 for PIL images automatically.
# As no explicit transforms.Resize was in your load_data, we assume images are either
# pre-sized or the model is robust to slight variations due to AdaptiveAvgPool2d.
# However, for consistency and common CNN practice, a Resize is included.
# IMPORTANT: Adjust IMAGE_TARGET_SIZE if your training images had a specific fixed dimension.
IMAGE_TARGET_SIZE = (224, 224) # Common for CNNs. Adjust if your MRI images were different (e.g., 128x128, 256x256)

preprocess = transforms.Compose([
    transforms.Resize(IMAGE_TARGET_SIZE), # Resize image to a consistent input size
    transforms.ToTensor(),                # Converts PIL image to FloatTensor and scales pixel values from [0, 255] to [0.0, 1.0]
    # IMPORTANT: Your newtraining.py did NOT explicitly include transforms.Normalize with mean/std.
    # If your model was trained WITHOUT ImageNet-style normalization, KEEP THIS LINE COMMENTED OUT.
    # If you used it (e.g., with transfer learning from a pre-trained model), UNCOMMENT and use appropriate mean/std.
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Standard ImageNet normalization
])

async def predict_image(image_bytes: bytes):
    """
    Performs inference on an uploaded MRI image using the loaded deep learning model.

    Args:
        image_bytes (bytes): The raw bytes of the uploaded image file.

    Returns:
        tuple: (predicted_class_name, confidence_percentage, all_probabilities_dict)
    """
    model = load_model() # Ensure the model is loaded before prediction

    try:
        # Open image from bytes, convert to RGB (ensures 3 channels for consistency with model input)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB") 
        
        # Apply the defined preprocessing transformations to the image
        input_tensor = preprocess(image)
        # Add a batch dimension to the tensor (models expect input in BxCxHxW format)
        input_batch = input_tensor.unsqueeze(0) 

        # Perform inference with no gradient calculation (saves memory and speeds up inference)
        with torch.no_grad(): 
            # Move the input tensor to the same device (CPU/GPU) as the model
            output = model(input_batch.to(device))
            # Apply softmax to convert raw model outputs (logits) into probabilities
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get the index of the class with the highest probability
            predicted_class_idx = torch.argmax(probabilities).item()
            # Map the index to the human-readable class name
            predicted_class_name = CLASS_NAMES[predicted_class_idx]
            # Calculate confidence as a percentage
            confidence = probabilities[predicted_class_idx].item() * 100

            # Prepare a dictionary of all class probabilities for the diagnostic report
            all_probabilities = {name: prob.item() * 100 for name, prob in zip(CLASS_NAMES, probabilities)}

        return predicted_class_name, confidence, all_probabilities
    except Exception as e:
        print(f"Error during image prediction: {e}")
        # Re-raise the exception to be caught by the FastAPI error handler
        raise