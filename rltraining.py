import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import logging
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("rl_nas_training.log"), logging.StreamHandler()]
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Base directory for data
# Use project root as base directory so Training/Testing folders in the repo are found
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))

# Function to load data with augmentation
def load_data(data_type):
    data = []
    labels = []
    label_map = {"glioma_tumor": 0, "meningioma_tumor": 1, "no_tumor": 2, "pituitary_tumor": 3}

    # Make sure train and eval use consistent preprocessing (resize + normalization).
    IMAGE_SIZE = (224, 224)
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_eval = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
                    # Apply the appropriate transform and convert back to numpy (C, H, W)
                    if data_type == "Training":
                        tensor_sample = transform_train(sample)
                    else:
                        tensor_sample = transform_eval(sample)
                    sample = tensor_sample.numpy()
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

# No transpose needed - shape should be (samples, channels, height, width)
logging.info(f"Reshaped X_train shape: {X_train.shape}")

# Controller (RNN) to generate architectures
class Controller(nn.Module):
    def __init__(self, hidden_size=64):
        super(Controller, self).__init__()
        self.rnn = nn.RNNCell(1, hidden_size)
        self.fc_filters = nn.Linear(hidden_size, 4)  # [16, 32, 64, 128]
        self.fc_kernels = nn.Linear(hidden_size, 2)  # [2, 5]
        self.fc_activations = nn.Linear(hidden_size, 2)  # [relu, leakyrelu]
        self.fc_layers = nn.Linear(hidden_size, 2)  # [2, 3] layers

    def forward(self, inputs, hidden):
        hidden = self.rnn(inputs, hidden)
        filters = F.softmax(self.fc_filters(hidden), dim=-1)
        kernels = F.softmax(self.fc_kernels(hidden), dim=-1)
        activations = F.softmax(self.fc_activations(hidden), dim=-1)
        layers = F.softmax(self.fc_layers(hidden), dim=-1)
        return filters, kernels, activations, layers, hidden

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

# RL NAS training
logging.info("Starting RL-Based Neural Architecture Search...")
controller = Controller().to(device)
controller_optimizer = optim.Adam(controller.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

best_accuracy = 0
best_architecture = None
num_architectures = 5
num_epochs_per_arch = 3

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    train_dataset = TensorDataset(torch.from_numpy(X_train_fold), torch.from_numpy(y_train_fold))
    val_dataset = TensorDataset(torch.from_numpy(X_val_fold), torch.from_numpy(y_val_fold))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Reduced batch size
    val_loader = DataLoader(val_dataset, batch_size=16)

    hidden = torch.zeros(1, 64).to(device)
    for i in range(num_architectures):
        # Sample architecture
        inputs = torch.zeros(1, 1).to(device)  # Dummy input for RNN
        filters, kernels, activations, layers, hidden = controller(inputs, hidden)
        # Use Categorical distributions to sample and keep log-probs for REINFORCE
        filt_dist = torch.distributions.Categorical(probs=filters)
        kern_dist = torch.distributions.Categorical(probs=kernels)
        act_dist = torch.distributions.Categorical(probs=activations)
        layer_dist = torch.distributions.Categorical(probs=layers)

        sampled_layer = layer_dist.sample().item()
        num_layers = sampled_layer + 2  # map [0,1] -> [2,3]
        conv_layers = []
        log_probs = []
        for _ in range(num_layers):
            filter_idx = filt_dist.sample().item()
            kernel_idx = kern_dist.sample().item()
            act_idx = act_dist.sample().item()
            log_probs.append(filt_dist.log_prob(torch.tensor(filter_idx)))
            log_probs.append(kern_dist.log_prob(torch.tensor(kernel_idx)))
            log_probs.append(act_dist.log_prob(torch.tensor(act_idx)))
            filters_val = [16, 32, 64, 128][filter_idx]
            kernels_val = [2, 5][kernel_idx]
            act_val = ['relu', 'leakyrelu'][act_idx]
            conv_layers.append({'filters': filters_val, 'kernel_size': kernels_val, 'activation': act_val})
        architecture = {'conv_layers': conv_layers, 'fc_layers': [{'units': 256, 'activation': 'relu'}], 'dropout_rate': 0.5}

        # Train and evaluate model
        model = NASModel(architecture, num_classes=4).to(device)
        model_optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        for epoch in range(num_epochs_per_arch):
            train_loss = train(model, train_loader, criterion, model_optimizer, device)
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            logging.info(f"Fold {fold+1}, Arch {i+1}/{num_architectures}, Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Reward and update controller using collected log_probs
    reward = val_accuracy / 100.0  # Normalize to [0, 1]
    # Sum log-probs of the sampled actions
    log_prob_sum = torch.stack(log_probs).sum()
    loss = -log_prob_sum * reward
    controller_optimizer.zero_grad()
    loss.backward()
    controller_optimizer.step()

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_architecture = architecture.copy()

logging.info(f"Best architecture found: {best_architecture}")
logging.info(f"Best validation accuracy: {best_accuracy:.2f}%")

# Train the best architecture with early stopping
logging.info("Training best architecture on full training data with early stopping...")
full_train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
full_train_loader = DataLoader(full_train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=16)

best_model = NASModel(best_architecture, num_classes=4).to(device)
optimizer = optim.Adam(best_model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

best_loss = float('inf')
patience = 5
trigger_times = 0
best_model_state = None

for epoch in range(30):
    train_loss = train(best_model, full_train_loader, criterion, optimizer, device)
    val_loss, _ = evaluate(best_model, test_loader, criterion, device)
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
    torch.save(best_model.state_dict(), "best_model.pth")
    logging.info("Model saved as best_model.pth")

test_loss, test_accuracy = evaluate(best_model, test_loader, criterion, device)
logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    pass