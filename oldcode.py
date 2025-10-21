import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from PIL import Image
import logging
from sklearn.model_selection import train_test_split
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("nas_training.log"), logging.StreamHandler()]
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Base directory for data (update this path as needed)
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

# Reshape data to (samples, channels, height, width)
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))
logging.info(f"Reshaped X_train shape: {X_train.shape}")

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
logging.info(f"Data prepared: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

# Define activation map
activation_map = {
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU
}

# Generate random architecture
def generate_random_architecture():
    num_conv_layers = random.randint(2, 4)
    conv_layers = []
    for _ in range(num_conv_layers):
        filters = random.choice([16, 32, 64, 128])
        kernel_size = random.choice([3, 5])
        activation = random.choice(['relu', 'leakyrelu'])
        conv_layers.append({'filters': filters, 'kernel_size': kernel_size, 'activation': activation})
    
    num_fc_layers = random.randint(1, 2)
    fc_layers = []
    for _ in range(num_fc_layers):
        units = random.choice([128, 256, 512])
        activation = random.choice(['relu', 'leakyrelu'])
        fc_layers.append({'units': units, 'activation': activation})
    
    dropout_rate = random.choice([0.0, 0.2, 0.5])
    
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
        x = x.view(x.size(0), -1)  # Flatten
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

# Perform NAS with random search
logging.info("Starting Neural Architecture Search...")
best_accuracy = 0
best_architecture = None
num_architectures = 10  # Number of architectures to try

for i in range(num_architectures):
    architecture = generate_random_architecture()
    model = NASModel(architecture, num_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train for a few epochs
    for epoch in range(5):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        logging.info(f"Arch {i+1}/{num_architectures}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
    
    # Evaluate on validation set
    _, val_accuracy = evaluate(model, val_loader, criterion, device)
    logging.info(f"Arch {i+1}/{num_architectures}, Validation Accuracy: {val_accuracy:.2f}%")
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_architecture = architecture

logging.info(f"Best architecture found: {best_architecture}")
logging.info(f"Best validation accuracy: {best_accuracy:.2f}%")

# Optional: Train the best architecture fully and test it
logging.info("Training best architecture on full training data...")
full_train_dataset = TensorDataset(torch.from_numpy(np.concatenate((X_train, X_val))), 
                                  torch.from_numpy(np.concatenate((y_train, y_val))))
full_train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=True)

best_model = NASModel(best_architecture, num_classes=4).to(device)
optimizer = optim.Adam(best_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(7):  # Train for more epochs
    train_loss = train(best_model, full_train_loader, criterion, optimizer, device)
    logging.info(f"Best Model, Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

test_loss, test_accuracy = evaluate(best_model, test_loader, criterion, device)
logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")