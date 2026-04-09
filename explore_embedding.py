#!/usr/bin/env python
# coding: utf-8


import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub import login
from transformers import AutoModel

# ============================================
# CONFIGURATION
# ============================================
class Config:
    dataset_path = "data"
    hf_token = "[Huggingface_token]"
    reve_model = "brain-bzh/reve-large"
    electrode_names = ['F3', 'F7', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8']
    sample_rate = 200
    batch_size = 512
    num_epochs = 500
    patience = 5
    learning_rate = 0.00001
    random_seed = 42

# ============================================
# DATA LOADING
# ============================================
def load_eeg_data(config):
    """Load and preprocess EEG data from .fif files."""
    X, y = [], []
    
    for root, dirs, files in os.walk(config.dataset_path):
        for file in files:
            if not file.endswith(".fif"):
                continue
                
            file_path = os.path.join(root, file)
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)
            epochs = epochs.copy().resample(config.sample_rate, npad='auto')
            epochs = epochs.pick(config.electrode_names)
            
            data = epochs.get_data()
            labels = epochs.metadata['p_cloze'].values
            
            X.extend(data[i] for i in range(len(data)))
            y.extend(labels[i])
    
    X = np.stack(X)
    y = np.array(y)
    
    # Remove NaN values
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask] * 100
    
    print(f"Data loaded: X shape={X.shape}, y shape={y.shape}")
    return X, y

# ============================================
# REVE MODEL FEATURE EXTRACTION
# ============================================
def extract_reve_features(eeg_data, electrode_names, config):
    """Extract features using REVE model."""
    print("Loading REVE models...")
    
    # Login to HuggingFace
    login(token=config.hf_token)
    
    # Load models
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
    model = AutoModel.from_pretrained(config.reve_model, trust_remote_code=True)
    model.eval()
    
    # Prepare data
    eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
    positions = pos_bank(electrode_names)
    positions = positions.expand(eeg_tensor.size(0), -1, -1)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Extract features in batches
    dataset = TensorDataset(eeg_tensor, positions)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    outputs = []
    with torch.no_grad():
        for batch_eeg, batch_positions in dataloader:
            batch_eeg = batch_eeg.to(device)
            batch_positions = batch_positions.to(device)
            batch_output = model(batch_eeg, batch_positions)
            outputs.append(batch_output.detach().cpu())
            
            # Clean up GPU memory
            del batch_eeg, batch_positions, batch_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    features = torch.cat(outputs, dim=0).squeeze()
    print(f"REVE features extracted: shape={features.shape}")
    return features

# ============================================
# MODEL DEFINITION
# ============================================
class EEGConvNet(nn.Module):
    """CNN for EEG regression task."""
    
    def __init__(self, input_shape=(12, 6, 512), num_classes=1):
        super().__init__()
        self.C, self.H, self.W = input_shape
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(self.C, 32, kernel_size=(3, 5), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 4))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(2, 5), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((1, 4))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 32))
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 32, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Handle various input shapes
        if len(x.shape) == 5:
            x = x.squeeze(1)
        elif len(x.shape) == 3:
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.view(x.size(0), self.C, self.H, self.W)
        
        # Forward pass
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x.squeeze(-1)

# ============================================
# TRAINING UTILITIES
# ============================================
def train_model(model, train_loader, val_loader, config):
    """Train the model with early stopping."""
    device = next(model.parameters()).device
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_y = batch_y.view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_y = batch_y.view(-1, 1)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= config.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return model

def evaluate_model(model, test_loader, tolerance=10):
    """Evaluate model and print metrics."""
    device = next(model.parameters()).device
    model.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            
            preds = outputs.squeeze(1) if outputs.dim() == 2 else outputs.flatten()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    abs_differences = np.abs(all_preds - all_labels)
    within_tolerance = abs_differences <= tolerance
    
    print("=" * 50)
    print(f"MODEL EVALUATION (tolerance: ±{tolerance})")
    print("=" * 50)
    print(f"Total samples: {len(all_labels)}")
    print(f"Within tolerance: {within_tolerance.sum()} ({within_tolerance.mean()*100:.2f}%)")
    print(f"Outside tolerance: {(~within_tolerance).sum()} ({100 - within_tolerance.mean()*100:.2f}%)")
    print("-" * 50)
    print(f"Mean Absolute Error: {abs_differences.mean():.4f}")
    print(f"Max Error: {abs_differences.max():.4f}")
    print(f"Std Error: {abs_differences.std():.4f}")
    print("=" * 50)
    
    return all_preds, all_labels



# In[10]:


#!/usr/bin/env python
# coding: utf-8

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub import login
from transformers import AutoModel

# ============================================
# MODEL ARCHITECTURE
# ============================================
class EEGConvNet(nn.Module):
    """CNN for EEG regression task."""
    
    def __init__(self, input_channels, input_height, input_width, num_classes=1):
        super().__init__()
        self.C, self.H, self.W = input_channels, input_height, input_width
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(self.C, 32, kernel_size=(3, 5), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 4))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(2, 5), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((1, 4))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 32))
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 32, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Handle various input shapes
        if len(x.shape) == 5:
            x = x.squeeze(1)
        elif len(x.shape) == 3:
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.view(x.size(0), self.C, self.H, self.W)
        
        # Forward pass
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x.squeeze(-1)

# ============================================
# DATA LOADING FUNCTIONS
# ============================================
def load_eeg_data(dataset_path, electrode_names, sample_rate):
    """Load and preprocess EEG data from .fif files."""
    X, y = [], []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if not file.endswith(".fif"):
                continue
                
            file_path = os.path.join(root, file)
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)
            epochs = epochs.copy().resample(sample_rate, npad='auto')
            epochs = epochs.pick(electrode_names)
            
            data = epochs.get_data()
            labels = epochs.metadata['p_cloze'].values
            
            # Fixed: Use proper loop instead of undefined variable 'i'
            for i in range(len(data)):
                X.append(data[i])
                y.append(labels[i])
    
    X = np.stack(X)
    y = np.array(y)
    
    # Remove NaN values
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask] * 100
    
    print(f"Data loaded: X shape={X.shape}, y shape={y.shape}")
    return X, y

def extract_reve_features(eeg_data, electrode_names, hf_token, reve_model, batch_size):
    """Extract features using REVE model."""
    print("Loading REVE models...")
    
    # Login to HuggingFace
    login(token=hf_token)
    
    # Load models
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
    model = AutoModel.from_pretrained(reve_model, trust_remote_code=True)
    model.eval()
    
    # Prepare data
    eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
    positions = pos_bank(electrode_names)
    positions = positions.expand(eeg_tensor.size(0), -1, -1)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Extract features in batches
    dataset = TensorDataset(eeg_tensor, positions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    outputs = []
    with torch.no_grad():
        for batch_eeg, batch_positions in dataloader:
            batch_eeg = batch_eeg.to(device)
            batch_positions = batch_positions.to(device)
            batch_output = model(batch_eeg, batch_positions)
            outputs.append(batch_output.detach().cpu())
            
            # Clean up GPU memory
            del batch_eeg, batch_positions, batch_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    features = torch.cat(outputs, dim=0).squeeze()
    print(f"REVE features extracted: shape={features.shape}")
    return features

# ============================================
# TRAINING FUNCTIONS
# ============================================
def train_model(model, train_loader, val_loader, learning_rate, num_epochs, patience):
    """Train the model with early stopping."""
    device = next(model.parameters()).device
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_y = batch_y.view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_y = batch_y.view(-1, 1)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return model

def evaluate_model(model, test_loader, tolerance):
    """Evaluate model and print metrics."""
    device = next(model.parameters()).device
    model.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            
            preds = outputs.squeeze(1) if outputs.dim() == 2 else outputs.flatten()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    abs_differences = np.abs(all_preds - all_labels)
    within_tolerance = abs_differences <= tolerance
    
    print("=" * 50)
    print(f"MODEL EVALUATION (tolerance: ±{tolerance})")
    print("=" * 50)
    print(f"Total samples: {len(all_labels)}")
    print(f"Within tolerance: {within_tolerance.sum()} ({within_tolerance.mean()*100:.2f}%)")
    print(f"Outside tolerance: {(~within_tolerance).sum()} ({100 - within_tolerance.mean()*100:.2f}%)")
    print("-" * 50)
    print(f"Mean Absolute Error: {abs_differences.mean():.4f}")
    print(f"Max Error: {abs_differences.max():.4f}")
    print(f"Std Error: {abs_differences.std():.4f}")
    print("=" * 50)
    
    return all_preds, all_labels



# In[12]:


# ============================================
# MAIN PIPELINE
# ============================================

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# ========================================
# ALL PARAMETERS - CONFIGURE HERE
# ========================================

# Data parameters
DATASET_PATH = "../Comprehension/Derco/"
ELECTRODE_NAMES = ['F3', 'F7', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8']
SAMPLE_RATE = 200

# REVE model parameters
HF_TOKEN = ""
REVE_MODEL = "brain-bzh/reve-large"  # or "brain-bzh/reve-base" for smaller model
REVE_BATCH_SIZE = 512

# Model architecture parameters
INPUT_CHANNELS = 1  # Added channel dimension
INPUT_HEIGHT = 12   # Number of electrodes
INPUT_WIDTH = 512   # Feature dimension from REVE

# Training parameters
BATCH_SIZE = 512
LEARNING_RATE = 0.00001
NUM_EPOCHS = 500
PATIENCE = 10
EVAL_TOLERANCE = 10

# Split parameters
TEST_SIZE = 0.4
VAL_SIZE = 0.5  # 50% of temp (20% of total)
RANDOM_SEED = 42

# Device parameters
USE_MULTI_GPU = True  # Automatically use all available GPUs if True

# Print configuration
print("=" * 50)
print("CONFIGURATION")
print("=" * 50)
print(f"Dataset path: {DATASET_PATH}")
print(f"Electrodes: {len(ELECTRODE_NAMES)} channels")
print(f"Sample rate: {SAMPLE_RATE} Hz")
print(f"REVE model: {REVE_MODEL}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Max epochs: {NUM_EPOCHS}")
print(f"Early stopping patience: {PATIENCE}")
print(f"Random seed: {RANDOM_SEED}")
print("=" * 50)

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load data
print("\n" + "="*50)
print("STEP 1: Loading EEG data")
print("="*50)
X, y = load_eeg_data(DATASET_PATH, ELECTRODE_NAMES, SAMPLE_RATE)

# Extract REVE features
print("\n" + "="*50)
print("STEP 2: Extracting REVE features")
print("="*50)
features = extract_reve_features(X, ELECTRODE_NAMES, HF_TOKEN, REVE_MODEL, REVE_BATCH_SIZE)

# Train/validation/test split
print("\n" + "="*50)
print("STEP 3: Splitting dataset")
print("="*50)
X_train, X_temp, y_train, y_temp = train_test_split(
    features, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_SEED
)

print(f"Train: {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")

# Convert to tensors and add channel dimension
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Setup device and model
print("\n" + "="*50)
print("STEP 4: Setting up model")
print("="*50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Initialize model
model = EEGConvNet(INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

# Multi-GPU setup
if USE_MULTI_GPU and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = nn.DataParallel(model)

model = model.to(device)

# Print model summary (optional)
try:
    from torchsummary import summary
    summary(model, input_size=(INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH))
except:
    print("Model summary not available (torchsummary not installed)")

# Train model
print("\n" + "="*50)
print("STEP 5: Training model")
print("="*50)
model = train_model(model, train_loader, val_loader, LEARNING_RATE, NUM_EPOCHS, PATIENCE)

# Load best model
model.load_state_dict(torch.load("best_model.pth", map_location=device))

# Evaluate model
print("\n" + "="*50)
print("STEP 6: Evaluating model")
print("="*50)
evaluate_model(model, test_loader, EVAL_TOLERANCE)

print("\n" + "="*50)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("="*50)





