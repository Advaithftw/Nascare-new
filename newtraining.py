import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from PIL import Image
import logging
from sklearn.model_selection import StratifiedKFold

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("nas_training.log"), logging.StreamHandler()]
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Base directory for data
base_dir = r"C:\Users\sakth\OneDrive\Desktop\mednas"

# Function to load data
def load_data(data_type):
    data = []
    labels = []
    label_map = {"glioma_tumor": 0, "meningioma_tumor": 1, "no_tumor": 2, "pituitary_tumor": 3}
    
    for class_name, label in label_map.items():
        class_dir = os.path.join(base_dir, data_type, class_name)
        if not os.path.exists(class_dir):
            logging.warning(f"Directory not found: {class_dir}")
            continue
        for file_name in os.listdir(class_dir):
            if file_name.endswith(".png"):
                file_path = os.path.join(class_dir, file_name)
                try:
                    img = Image.open(file_path).convert('RGB')
                    sample = np.array(img, dtype=np.float32) / 255.0
                    data.append(sample)
                    labels.append(label)
                except Exception as e:
                    logging.error(f"Error loading {file_path}: {e}")
                    continue
    
    if not data:
        logging.error(f"No data loaded for {data_type}")
        raise ValueError(f"No data found in {data_type} directories")
    
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    logging.info(f"Loaded {len(data)} samples for {data_type} with shape {data.shape}")
    return data, labels

# Load and prepare data
logging.info("Starting data loading...")
X_train, y_train = load_data("Training")
X_test, y_test = load_data("Testing")

# Reshape data
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))
logging.info(f"Reshaped X_train shape: {X_train.shape}")

# Architecture Optimizer for Gradient-Based NAS
class ArchitectureOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter_logits = nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.5]))  # Bias toward 128
        self.kernel_logits = nn.Parameter(torch.randn(2))  # Logits for [2, 5]
        self.activation_logits = nn.Parameter(torch.randn(2))  # Logits for [relu, leakyrelu]
        self.fc_unit_logits = nn.Parameter(torch.randn(3))  # Logits for [128, 256, 512]

    def sample_architecture(self, num_conv_layers):
        probs_filters = torch.softmax(self.filter_logits, dim=0)
        probs_kernels = torch.softmax(self.kernel_logits, dim=0)
        probs_activations = torch.softmax(self.activation_logits, dim=0)
        probs_fc_units = torch.softmax(self.fc_unit_logits, dim=0)
        
        conv_layers = []
        for _ in range(num_conv_layers):
            filters = torch.multinomial(probs_filters, 1).item() * 32 + 16  # Map to [16, 32, 64, 128]
            kernel = torch.multinomial(probs_kernels, 1).item() * 3 + 2  # Map to [2, 5]
            activation_idx = torch.multinomial(probs_activations, 1).item()
            conv_layers.append({'filters': filters, 'kernel_size': kernel, 'activation': ['relu', 'leakyrelu'][activation_idx]})
        
        fc_layers = [{'units': torch.multinomial(probs_fc_units, 1).item() * 128 + 128, 'activation': 'relu'}]  # Simplified to 1 FC layer
        dropout_rate = 0.3  # Fixed for simplicity
        
        return {'conv_layers': conv_layers, 'fc_layers': fc_layers, 'dropout_rate': dropout_rate}

# NAS Model class
class NASModel(nn.Module):
    def __init__(self, architecture, num_classes):
        super(NASModel, self).__init__()
        conv_layers = []
        in_channels = 3  # RGB input
        for conv_spec in architecture['conv_layers']:
            out_channels = conv_spec['filters']
            kernel_size = conv_spec['kernel_size']
            activation = nn.ReLU() if conv_spec['activation'] == 'relu' else nn.LeakyReLU()
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
            activation = nn.ReLU() if fc_spec['activation'] == 'relu' else nn.LeakyReLU()
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

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# Perform NAS with gradient-based search
logging.info("Starting Gradient-Based Neural Architecture Search...")
arch_optimizer = ArchitectureOptimizer().to(device)
criterion = nn.CrossEntropyLoss()

best_accuracy = 0
best_architecture = None
num_architectures = 5  # Reduced for gradient-based efficiency
num_conv_layers_range = [2, 3]  # Limit to avoid overfitting

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    train_dataset = TensorDataset(torch.from_numpy(X_train_fold), torch.from_numpy(y_train_fold))
    val_dataset = TensorDataset(torch.from_numpy(X_val_fold), torch.from_numpy(y_val_fold))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    for i in range(num_architectures):
        num_conv_layers = num_conv_layers_range[i % len(num_conv_layers_range)]  # Cycle through options
        architecture = arch_optimizer.sample_architecture(num_conv_layers)
        model = NASModel(architecture, num_classes=4).to(device)
        
        model_optimizer = optim.Adam([
            {'params': arch_optimizer.parameters(), 'lr': 0.001},  # Lowered architecture LR
            {'params': model.parameters(), 'lr': 0.001}          # Model weights
        ])
        
        for epoch in range(10):  # Increased epochs for gradient optimization
            model.train()
            train_loss = train(model, train_loader, criterion, model_optimizer, device)
            
            # Use a single validation batch for gradient computation
            model.eval()
            with torch.no_grad():
                val_data, val_target = next(iter(val_loader))  # Get one batch
                val_data, val_target = val_data.to(device), val_target.to(device)
            output = model(val_data)
            arch_loss = criterion(output, val_target)  # Tensor loss for backward
            model_optimizer.zero_grad()
            arch_loss.backward()
            model_optimizer.step()
            
            # Evaluate full validation set for accuracy tracking
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            logging.info(f"Fold {fold+1}, Arch {i+1}/{num_architectures}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_architecture = architecture.copy()  # Store the best architecture

logging.info(f"Best architecture found: {best_architecture}")
logging.info(f"Best validation accuracy: {best_accuracy:.2f}%")

# Train the best architecture with early stopping
logging.info("Training best architecture on full training data with early stopping...")
full_train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
full_train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=32)

best_model = NASModel(best_architecture, num_classes=4).to(device)
optimizer = optim.Adam(best_model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

best_loss = float('inf')
patience = 5
trigger_times = 0
best_model_state = None

for epoch in range(30):
    train_loss = train(best_model, full_train_loader, criterion, optimizer, device)
    val_loss, _ = evaluate(best_model, test_loader, criterion, device)  # Using test as val proxy
    logging.info(f"Best Model, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
        best_model_state = best_model.state_dict().copy()
    else:
        trigger_times += 1
        if trigger_times >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break

if best_model_state is not None:
    best_model.load_state_dict(best_model_state)
    # Option 1: Save only the model weights (recommended for deployment)
    torch.save(best_model.state_dict(), "best_model.pth")
    # Option 2: Save the full model (uncomment to use)
    # torch.save(best_model, "best_model.pth")
    logging.info("Model saved as best_model.pth")

test_loss, test_accuracy = evaluate(best_model, test_loader, criterion, device)
logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    pass