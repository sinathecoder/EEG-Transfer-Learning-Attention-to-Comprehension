#!/usr/bin/env python
# coding: utf-8

import os
import gc
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub import login
from transformers import AutoModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ============================================
# CONFIGURATION
# ============================================
class Config:
    # Data parameters
    dataset_path = "../Comprehension/Derco/"
    electrode_names = ['F3', 'F7', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8']
    sample_rate = 200
    
    # REVE model parameters
    hf_token = "[Huggingface_token]"
    reve_model = "brain-bzh/reve-large"  # or "brain-bzh/reve-base"
    reve_batch_size = 512
    
    # Training parameters
    batch_size = 512
    learning_rate = 0.00001
    num_epochs = 500
    patience = 10
    eval_tolerance = 10
    
    # Participant split parameters
    n_val_participants = 2
    n_test_participants = 5
    random_seed = 42
    
    # Model architecture
    input_channels = 1
    input_height = 12  # Number of electrodes
    input_width = 512  # REVE feature dimension
    
    # Device
    use_multi_gpu = True

# ============================================
# UTILITIES
# ============================================
def reset_weights(m):
    """Reset model weights for training from scratch."""
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def extract_subject_id_from_path(file_path):
    """Extract participant ID from the folder structure."""
    parts = file_path.split(os.sep)
    
    for i, part in enumerate(parts):
        if re.match(r'^[A-Z]{3}\d{2}$', part):
            return part
        if re.match(r'^[A-Z]{3}\d{2}$', part):
            return part
    
    for i, part in enumerate(parts):
        if part.startswith('article_') and i > 0:
            return parts[i-1]
    
    return "unknown"

# ============================================
# DATA LOADING WITH PARTICIPANT INFO
# ============================================
def load_dataset_with_participants(dataset_path, electrode_names, sample_rate):
    """Load dataset with participant/subject information."""
    X, y, subjects = [], [], []
    subject_counter = defaultdict(int)
    subject_articles = defaultdict(set)
    
    fif_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".fif"):
                fif_files.append(os.path.join(root, file))
    
    print(f"Found {len(fif_files)} .fif files")
    
    for file_path in fif_files:
        # Extract participant ID
        subject_id = extract_subject_id_from_path(file_path)
        
        # Extract article number
        article_match = re.search(r'article[_]?(\d+)', file_path)
        article = article_match.group(1) if article_match else "unknown"
        
        subject_counter[subject_id] += 1
        subject_articles[subject_id].add(article)
        
        try:
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)
            epochs = epochs.copy().resample(sample_rate, npad='auto')
            epochs = epochs.pick(electrode_names)
            
            # Check if metadata has p_cloze column
            if 'p_cloze' not in epochs.metadata.columns:
                print(f"Warning: No p_cloze in {file_path}, skipping...")
                continue
                
            data = epochs.get_data()
            labels = epochs.metadata['p_cloze'].values
            
            for i in range(len(data)):
                if not np.isnan(labels[i]):
                    X.append(data[i])
                    y.append(labels[i])
                    subjects.append(subject_id)
                    
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if len(X) == 0:
        raise ValueError("No valid data loaded. Check if files have 'p_cloze' metadata.")
    
    X = np.stack(X)
    y = np.array(y) * 100  # scale to percentage
    subjects = np.array(subjects)
    
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {len(X)}")
    print(f"Unique participants: {len(np.unique(subjects))}")
    print(f"\nParticipants found:")
    for subj in sorted(np.unique(subjects)):
        n_samples = np.sum(subjects == subj)
        n_articles = len(subject_articles[subj])
        print(f"  {subj}: {n_samples} samples from {n_articles} articles")
    
    return X, y, subjects

def participant_based_split(X, y, subjects, n_val_participants=2, n_test_participants=5, random_state=42):
    """Split data based on participants with explicit control over participant counts."""
    np.random.seed(random_state)
    
    # Get unique participants
    unique_subjects = np.unique(subjects)
    n_total = len(unique_subjects)
    
    print(f"\n{'='*60}")
    print("PARTICIPANT-BASED SPLIT")
    print(f"{'='*60}")
    print(f"Total participants: {n_total}")
    print(f"Participants: {sorted(unique_subjects)}")
    
    # Validate counts
    min_required = n_val_participants + n_test_participants + 1
    if n_total < min_required:
        raise ValueError(f"Need at least {min_required} participants, but only have {n_total}")
    
    if n_val_participants + n_test_participants >= n_total:
        print(f"Warning: Adjusting test participants. Total={n_total}, val={n_val_participants}")
        n_test_participants = n_total - n_val_participants - 1
        print(f"Adjusted test participants to {n_test_participants}")
    
    # Randomly select validation participants
    val_subjects = np.random.choice(unique_subjects, size=n_val_participants, replace=False)
    remaining_subjects = [s for s in unique_subjects if s not in val_subjects]
    
    # Randomly select test participants from remaining
    test_subjects = np.random.choice(remaining_subjects, size=n_test_participants, replace=False)
    train_subjects = [s for s in remaining_subjects if s not in test_subjects]
    
    print(f"\nSplit Configuration:")
    print(f"  Validation participants ({len(val_subjects)}): {sorted(val_subjects)}")
    print(f"  Test participants ({len(test_subjects)}): {sorted(test_subjects)}")
    print(f"  Train participants ({len(train_subjects)}): {sorted(train_subjects)}")
    
    # Create masks for each split
    train_mask = np.isin(subjects, train_subjects)
    val_mask = np.isin(subjects, val_subjects)
    test_mask = np.isin(subjects, test_subjects)
    
    # Apply masks
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    subjects_train = subjects[train_mask]
    subjects_val = subjects[val_mask]
    subjects_test = subjects[test_mask]
    
    print(f"\nSample Distribution:")
    print(f"  Train: {len(X_train)} samples from {len(train_subjects)} participants")
    print(f"  Val:   {len(X_val)} samples from {len(val_subjects)} participants")
    print(f"  Test:  {len(X_test)} samples from {len(test_subjects)} participants")
    
    # Show samples per participant
    print(f"\nDetailed breakdown:")
    print(f"\nTRAIN SET:")
    for subj in sorted(train_subjects):
        count = np.sum(subjects[train_mask] == subj)
        print(f"    {subj}: {count} samples")
    
    print(f"\nVALIDATION SET:")
    for subj in sorted(val_subjects):
        count = np.sum(subjects[val_mask] == subj)
        print(f"    {subj}: {count} samples")
    
    print(f"\nTEST SET:")
    for subj in sorted(test_subjects):
        count = np.sum(subjects[test_mask] == subj)
        print(f"    {subj}: {count} samples")
    
    split_info = {
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
        'n_test_samples': len(X_test)
    }
    
    return X_train, X_val, X_test, y_train, y_val, y_test, subjects_train, subjects_val, subjects_test, split_info

# ============================================
# REVE FEATURE EXTRACTION
# ============================================
def extract_reve_features_for_splits(X_train, X_val, X_test, electrode_names, config):
    """Extract REVE features for each split separately."""
    print("\n" + "="*50)
    print("Extracting REVE features")
    print("="*50)
    
    # Login to HuggingFace
    login(token=config.hf_token)
    
    # Load models
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
    model = AutoModel.from_pretrained(config.reve_model, trust_remote_code=True)
    model.eval()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    def extract_features(data):
        """Extract REVE features for a dataset."""
        if len(data) == 0:
            return torch.tensor([])
        
        eeg_tensor = torch.tensor(data, dtype=torch.float32)
        positions = pos_bank(electrode_names)
        positions = positions.expand(eeg_tensor.size(0), -1, -1)
        
        dataset = TensorDataset(eeg_tensor, positions)
        dataloader = DataLoader(dataset, batch_size=config.reve_batch_size, shuffle=False)
        
        outputs = []
        with torch.no_grad():
            for batch_eeg, batch_positions in dataloader:
                batch_eeg = batch_eeg.to(device)
                batch_positions = batch_positions.to(device)
                batch_output = model(batch_eeg, batch_positions)
                outputs.append(batch_output.detach().cpu())
                
                del batch_eeg, batch_positions, batch_output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        features = torch.cat(outputs, dim=0).squeeze()
        return features
    
    # Extract features for each split
    print("Extracting training features...")
    features_train = extract_features(X_train)
    print(f"Training features shape: {features_train.shape}")
    
    print("Extracting validation features...")
    features_val = extract_features(X_val)
    print(f"Validation features shape: {features_val.shape}")
    
    print("Extracting test features...")
    features_test = extract_features(X_test)
    print(f"Test features shape: {features_test.shape}")
    
    return features_train, features_val, features_test

# ============================================
# MODEL DEFINITIONS
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

class NewHead(nn.Module):
    """New head for transfer learning."""
    def __init__(self, in_features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class REVERegressor(nn.Module):
    """Simple regressor on top of REVE features (Experiment 4)."""
    def __init__(self, input_dim=512, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ============================================
# TRAINING AND EVALUATION
# ============================================
def train_model(model, train_loader, val_loader, config, model_suffix=""):
    """Train the model with early stopping."""
    device = next(model.parameters()).device
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        scheduler.step(avg_val_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}/{config.num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model{model_suffix}.pth")
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f"best_model{model_suffix}.pth"))
    return model

def train_reve_regressor(model, features_train, y_train, features_val, y_val, config, model_suffix=""):
    """Train the REVE regressor (simple MLP)."""
    device = next(model.parameters()).device
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    
    # Create data loaders
    train_dataset = TensorDataset(features_train, y_train)
    val_dataset = TensorDataset(features_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
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
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}/{config.num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model{model_suffix}.pth")
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f"best_model{model_suffix}.pth"))
    return model

def evaluate_model(model, test_loader, config, split_info=None, suffix=""):
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
    within_tolerance = abs_differences <= config.eval_tolerance
    mae = np.mean(abs_differences)
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    mape = np.mean(np.abs((all_labels - all_preds) / (all_labels + 1e-8))) * 100
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS{suffix}")
    print(f"{'='*60}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Within tolerance (±{config.eval_tolerance}): {within_tolerance.sum()} ({within_tolerance.mean()*100:.2f}%)")
    
    if split_info:
        print(f"\nTest participants: {split_info['test_subjects']}")
        print(f"Test samples: {split_info['n_test_samples']}")
    
    return mae, rmse, mape

def evaluate_reve_regressor(model, features_test, y_test, config, split_info=None, suffix=""):
    """Evaluate REVE regressor model."""
    device = next(model.parameters()).device
    model.eval()
    
    features_test = features_test.to(device)
    y_test = y_test.to(device)
    
    with torch.no_grad():
        preds = model(features_test).cpu().numpy()
        labels = y_test.cpu().numpy()
    
    # Calculate metrics
    abs_differences = np.abs(preds - labels)
    within_tolerance = abs_differences <= config.eval_tolerance
    mae = np.mean(abs_differences)
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    mape = np.mean(np.abs((labels - preds) / (labels + 1e-8))) * 100
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS{suffix}")
    print(f"{'='*60}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Within tolerance (±{config.eval_tolerance}): {within_tolerance.sum()} ({within_tolerance.mean()*100:.2f}%)")
    
    if split_info:
        print(f"\nTest participants: {split_info['test_subjects']}")
        print(f"Test samples: {split_info['n_test_samples']}")
    
    return mae, rmse, mape

# ============================================
# EXPERIMENT RUNNER
# ============================================
def run_experiment_1_scratch(features_train, features_val, features_test, 
                              y_train, y_val, y_test, config, split_info):
    """Experiment 1: Train CNN from scratch on REVE features."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1: CNN FROM SCRATCH")
    print(f"{'='*60}")
    
    # Prepare data loaders (adding channel dimension for CNN)
    train_dataset = TensorDataset(features_train.unsqueeze(1), y_train)
    val_dataset = TensorDataset(features_val.unsqueeze(1), y_val)
    test_dataset = TensorDataset(features_test.unsqueeze(1), y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    model = EEGConvNet(config.input_channels, config.input_height, config.input_width)
    
    # Setup device and multi-GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Train model
    model = train_model(model, train_loader, val_loader, config, model_suffix="_exp1_scratch")
    
    # Evaluate
    mae, rmse, mape = evaluate_model(model, test_loader, config, split_info, suffix=" - EXP1: CNN FROM SCRATCH")
    
    return mae, rmse, mape

def run_experiment_2_transfer_unfrozen(features_train, features_val, features_test,
                                        y_train, y_val, y_test, config, split_info, pretrained_path):
    """Experiment 2: Transfer learning with unfrozen layers."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2: TRANSFER LEARNING (UNFROZEN)")
    print(f"{'='*60}")
    
    # Prepare data loaders
    train_dataset = TensorDataset(features_train.unsqueeze(1), y_train)
    val_dataset = TensorDataset(features_val.unsqueeze(1), y_val)
    test_dataset = TensorDataset(features_test.unsqueeze(1), y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    model = EEGConvNet(config.input_channels, config.input_height, config.input_width)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained weights
    loaded_model = torch.load(pretrained_path, map_location=device, weights_only=False)
    if isinstance(loaded_model, torch.nn.DataParallel):
        loaded_model = loaded_model.module
    
    # Load pretrained weights
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in loaded_model.state_dict().items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # Replace head
    if isinstance(model, torch.nn.DataParallel):
        model.module.fc3 = NewHead(64)
    else:
        model.fc3 = NewHead(64)
    
    # Multi-GPU setup
    if config.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Train model
    model = train_model(model, train_loader, val_loader, config, model_suffix="_exp2_unfrozen")
    
    # Evaluate
    mae, rmse, mape = evaluate_model(model, test_loader, config, split_info, suffix=" - EXP2: TRANSFER (UNFROZEN)")
    
    return mae, rmse, mape

def run_experiment_3_transfer_frozen(features_train, features_val, features_test,
                                      y_train, y_val, y_test, config, split_info, pretrained_path):
    """Experiment 3: Transfer learning with frozen layers."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 3: TRANSFER LEARNING (FROZEN)")
    print(f"{'='*60}")
    
    # Prepare data loaders
    train_dataset = TensorDataset(features_train.unsqueeze(1), y_train)
    val_dataset = TensorDataset(features_val.unsqueeze(1), y_val)
    test_dataset = TensorDataset(features_test.unsqueeze(1), y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    model = EEGConvNet(config.input_channels, config.input_height, config.input_width)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained weights
    loaded_model = torch.load(pretrained_path, map_location=device, weights_only=False)
    if isinstance(loaded_model, torch.nn.DataParallel):
        loaded_model = loaded_model.module
    
    # Load pretrained weights
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in loaded_model.state_dict().items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # Freeze all layers except the new head
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace head and unfreeze it
    if isinstance(model, torch.nn.DataParallel):
        model.module.fc3 = NewHead(64)
        for param in model.module.fc3.parameters():
            param.requires_grad = True
    else:
        model.fc3 = NewHead(64)
        for param in model.fc3.parameters():
            param.requires_grad = True
    
    # Multi-GPU setup
    if config.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Train model
    model = train_model(model, train_loader, val_loader, config, model_suffix="_exp3_frozen")
    
    # Evaluate
    mae, rmse, mape = evaluate_model(model, test_loader, config, split_info, suffix=" - EXP3: TRANSFER (FROZEN)")
    
    return mae, rmse, mape

def run_experiment_4_reve_baseline(features_train, features_val, features_test,
                                    y_train, y_val, y_test, config, split_info):
    """Experiment 4: REVE features + simple MLP regressor (baseline)."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 4: REVE BASELINE (MLP Regressor)")
    print(f"{'='*60}")
    
    # Initialize model
    model = REVERegressor(input_dim=features_train.shape[1])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Train model
    model = train_reve_regressor(model, features_train, y_train, features_val, y_val, 
                                  config, model_suffix="_exp4_reve")
    
    # Evaluate
    test_dataset = TensorDataset(features_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    mae, rmse, mape = evaluate_model(model, test_loader, config, split_info, suffix=" - EXP4: REVE BASELINE")
    
    return mae, rmse, mape

#!/usr/bin/env python
# coding: utf-8

import os
import gc
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub import login
from transformers import AutoModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler



# ============================================
# UTILITIES
# ============================================
def reset_weights(m):
    """Reset model weights for training from scratch."""
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def extract_subject_id_from_path(file_path):
    """Extract participant ID from the folder structure."""
    parts = file_path.split(os.sep)
    
    for i, part in enumerate(parts):
        if re.match(r'^[A-Z]{3}\d{2}$', part):
            return part
        if re.match(r'^[A-Z]{3}\d{2}$', part):
            return part
    
    for i, part in enumerate(parts):
        if part.startswith('article_') and i > 0:
            return parts[i-1]
    
    return "unknown"

# ============================================
# DATA LOADING WITH PARTICIPANT INFO
# ============================================
def load_dataset_with_participants(dataset_path, electrode_names, sample_rate):
    """Load dataset with participant/subject information."""
    X, y, subjects = [], [], []
    subject_counter = defaultdict(int)
    subject_articles = defaultdict(set)
    
    fif_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".fif"):
                fif_files.append(os.path.join(root, file))
    
    print(f"Found {len(fif_files)} .fif files")
    
    for file_path in fif_files:
        # Extract participant ID
        subject_id = extract_subject_id_from_path(file_path)
        
        # Extract article number
        article_match = re.search(r'article[_]?(\d+)', file_path)
        article = article_match.group(1) if article_match else "unknown"
        
        subject_counter[subject_id] += 1
        subject_articles[subject_id].add(article)
        
        try:
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)
            epochs = epochs.copy().resample(sample_rate, npad='auto')
            epochs = epochs.pick(electrode_names)
            
            # Check if metadata has p_cloze column
            if 'p_cloze' not in epochs.metadata.columns:
                print(f"Warning: No p_cloze in {file_path}, skipping...")
                continue
                
            data = epochs.get_data()
            labels = epochs.metadata['p_cloze'].values
            
            for i in range(len(data)):
                if not np.isnan(labels[i]):
                    X.append(data[i])
                    y.append(labels[i])
                    subjects.append(subject_id)
                    
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if len(X) == 0:
        raise ValueError("No valid data loaded. Check if files have 'p_cloze' metadata.")
    
    X = np.stack(X)
    y = np.array(y) * 100  # scale to percentage
    subjects = np.array(subjects)
    
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {len(X)}")
    print(f"Unique participants: {len(np.unique(subjects))}")
    print(f"\nParticipants found:")
    for subj in sorted(np.unique(subjects)):
        n_samples = np.sum(subjects == subj)
        n_articles = len(subject_articles[subj])
        print(f"  {subj}: {n_samples} samples from {n_articles} articles")
    
    return X, y, subjects

def participant_based_split(X, y, subjects, n_val_participants=2, n_test_participants=5, random_state=42):
    """Split data based on participants with explicit control over participant counts."""
    np.random.seed(random_state)
    
    # Get unique participants
    unique_subjects = np.unique(subjects)
    n_total = len(unique_subjects)
    
    print(f"\n{'='*60}")
    print("PARTICIPANT-BASED SPLIT")
    print(f"{'='*60}")
    print(f"Total participants: {n_total}")
    print(f"Participants: {sorted(unique_subjects)}")
    
    # Validate counts
    min_required = n_val_participants + n_test_participants + 1
    if n_total < min_required:
        raise ValueError(f"Need at least {min_required} participants, but only have {n_total}")
    
    if n_val_participants + n_test_participants >= n_total:
        print(f"Warning: Adjusting test participants. Total={n_total}, val={n_val_participants}")
        n_test_participants = n_total - n_val_participants - 1
        print(f"Adjusted test participants to {n_test_participants}")
    
    # Randomly select validation participants
    val_subjects = np.random.choice(unique_subjects, size=n_val_participants, replace=False)
    remaining_subjects = [s for s in unique_subjects if s not in val_subjects]
    
    # Randomly select test participants from remaining
    test_subjects = np.random.choice(remaining_subjects, size=n_test_participants, replace=False)
    train_subjects = [s for s in remaining_subjects if s not in test_subjects]
    
    print(f"\nSplit Configuration:")
    print(f"  Validation participants ({len(val_subjects)}): {sorted(val_subjects)}")
    print(f"  Test participants ({len(test_subjects)}): {sorted(test_subjects)}")
    print(f"  Train participants ({len(train_subjects)}): {sorted(train_subjects)}")
    
    # Create masks for each split
    train_mask = np.isin(subjects, train_subjects)
    val_mask = np.isin(subjects, val_subjects)
    test_mask = np.isin(subjects, test_subjects)
    
    # Apply masks
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    subjects_train = subjects[train_mask]
    subjects_val = subjects[val_mask]
    subjects_test = subjects[test_mask]
    
    print(f"\nSample Distribution:")
    print(f"  Train: {len(X_train)} samples from {len(train_subjects)} participants")
    print(f"  Val:   {len(X_val)} samples from {len(val_subjects)} participants")
    print(f"  Test:  {len(X_test)} samples from {len(test_subjects)} participants")
    
    # Show samples per participant
    print(f"\nDetailed breakdown:")
    print(f"\nTRAIN SET:")
    for subj in sorted(train_subjects):
        count = np.sum(subjects[train_mask] == subj)
        print(f"    {subj}: {count} samples")
    
    print(f"\nVALIDATION SET:")
    for subj in sorted(val_subjects):
        count = np.sum(subjects[val_mask] == subj)
        print(f"    {subj}: {count} samples")
    
    print(f"\nTEST SET:")
    for subj in sorted(test_subjects):
        count = np.sum(subjects[test_mask] == subj)
        print(f"    {subj}: {count} samples")
    
    split_info = {
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
        'n_test_samples': len(X_test)
    }
    
    return X_train, X_val, X_test, y_train, y_val, y_test, subjects_train, subjects_val, subjects_test, split_info

# ============================================
# REVE FEATURE EXTRACTION
# ============================================
def extract_reve_features_for_splits(X_train, X_val, X_test, electrode_names, config):
    """Extract REVE features for each split separately."""
    print("\n" + "="*50)
    print("Extracting REVE features")
    print("="*50)
    
    # Login to HuggingFace
    login(token=config.hf_token)
    
    # Load models
    pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
    model = AutoModel.from_pretrained(config.reve_model, trust_remote_code=True)
    model.eval()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    def extract_features(data):
        """Extract REVE features for a dataset."""
        if len(data) == 0:
            return torch.tensor([])
        
        eeg_tensor = torch.tensor(data, dtype=torch.float32)
        positions = pos_bank(electrode_names)
        positions = positions.expand(eeg_tensor.size(0), -1, -1)
        
        dataset = TensorDataset(eeg_tensor, positions)
        dataloader = DataLoader(dataset, batch_size=config.reve_batch_size, shuffle=False)
        
        outputs = []
        with torch.no_grad():
            for batch_eeg, batch_positions in dataloader:
                batch_eeg = batch_eeg.to(device)
                batch_positions = batch_positions.to(device)
                batch_output = model(batch_eeg, batch_positions)
                outputs.append(batch_output.detach().cpu())
                
                del batch_eeg, batch_positions, batch_output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        features = torch.cat(outputs, dim=0).squeeze()
        return features
    
    # Extract features for each split
    print("Extracting training features...")
    features_train = extract_features(X_train)
    print(f"Training features shape: {features_train.shape}")
    
    print("Extracting validation features...")
    features_val = extract_features(X_val)
    print(f"Validation features shape: {features_val.shape}")
    
    print("Extracting test features...")
    features_test = extract_features(X_test)
    print(f"Test features shape: {features_test.shape}")
    
    return features_train, features_val, features_test

# ============================================
# MODEL DEFINITIONS
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

class NewHead(nn.Module):
    """New head for transfer learning."""
    def __init__(self, in_features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class REVERegressor(nn.Module):
    """Simple regressor on top of REVE features (Experiment 4)."""
    def __init__(self, input_dim=512, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # Ensure x is 2D: (batch, features)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.net(x).squeeze(-1)

# ============================================
# TRAINING AND EVALUATION
# ============================================
def train_model(model, train_loader, val_loader, config, model_suffix=""):
    """Train the model with early stopping."""
    device = next(model.parameters()).device
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        scheduler.step(avg_val_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}/{config.num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model{model_suffix}.pth")
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f"best_model{model_suffix}.pth"))
    return model

def train_reve_regressor(model, features_train, y_train, features_val, y_val, config, model_suffix=""):
    """Train the REVE regressor (simple MLP)."""
    device = next(model.parameters()).device
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    
    # Print shapes for debugging
    print(f"Training data shape: {features_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {features_val.shape}")
    
    # Ensure features are 2D
    if len(features_train.shape) == 3:
        features_train = features_train.view(features_train.shape[0], -1)
        features_val = features_val.view(features_val.shape[0], -1)
        print(f"Reshaped training data to: {features_train.shape}")
    
    # Create data loaders
    train_dataset = TensorDataset(features_train.cpu(), y_train.cpu())
    val_dataset = TensorDataset(features_val.cpu(), y_val.cpu())
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
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
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}/{config.num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model{model_suffix}.pth")
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(f"best_model{model_suffix}.pth", map_location=device))
    return model    
def evaluate_model(model, test_loader, config, split_info=None, suffix=""):
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
    within_tolerance = abs_differences <= config.eval_tolerance
    mae = np.mean(abs_differences)
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
    mape = np.mean(np.abs((all_labels - all_preds) / (all_labels + 1e-8))) * 100
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS{suffix}")
    print(f"{'='*60}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Within tolerance (±{config.eval_tolerance}): {within_tolerance.sum()} ({within_tolerance.mean()*100:.2f}%)")
    
    if split_info:
        print(f"\nTest participants: {split_info['test_subjects']}")
        print(f"Test samples: {split_info['n_test_samples']}")
    
    return mae, rmse, mape

def evaluate_reve_regressor(model, features_test, y_test, config, split_info=None, suffix=""):
    """Evaluate REVE regressor model."""
    device = next(model.parameters()).device
    model.eval()
    
    # Ensure features are 2D
    if len(features_test.shape) == 3:
        features_test = features_test.view(features_test.shape[0], -1)
        print(f"Reshaped test data to: {features_test.shape}")
    
    # Move data to device
    features_test = features_test.to(device)
    y_test = y_test.to(device)
    
    with torch.no_grad():
        preds = model(features_test).cpu().numpy()
        labels = y_test.cpu().numpy()
    
    # Calculate metrics
    abs_differences = np.abs(preds - labels)
    within_tolerance = abs_differences <= config.eval_tolerance
    mae = np.mean(abs_differences)
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    mape = np.mean(np.abs((labels - preds) / (labels + 1e-8))) * 100
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS{suffix}")
    print(f"{'='*60}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Within tolerance (±{config.eval_tolerance}): {within_tolerance.sum()} ({within_tolerance.mean()*100:.2f}%)")
    
    if split_info:
        print(f"\nTest participants: {split_info['test_subjects']}")
        print(f"Test samples: {split_info['n_test_samples']}")
    
    return mae, rmse, mape
# ============================================
# EXPERIMENT RUNNER
# ============================================
def run_experiment_1_scratch(features_train, features_val, features_test, 
                              y_train, y_val, y_test, config, split_info):
    """Experiment 1: Train CNN from scratch on REVE features."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1: CNN FROM SCRATCH")
    print(f"{'='*60}")
    
    # Prepare data loaders (adding channel dimension for CNN)
    train_dataset = TensorDataset(features_train.unsqueeze(1), y_train)
    val_dataset = TensorDataset(features_val.unsqueeze(1), y_val)
    test_dataset = TensorDataset(features_test.unsqueeze(1), y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    model = EEGConvNet(config.input_channels, config.input_height, config.input_width)
    
    # Setup device and multi-GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Train model
    model = train_model(model, train_loader, val_loader, config, model_suffix="_exp1_scratch")
    
    # Evaluate
    mae, rmse, mape = evaluate_model(model, test_loader, config, split_info, suffix=" - EXP1: CNN FROM SCRATCH")
    
    return mae, rmse, mape

def run_experiment_2_transfer_unfrozen(features_train, features_val, features_test,
                                        y_train, y_val, y_test, config, split_info, pretrained_path):
    """Experiment 2: Transfer learning with unfrozen layers."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2: TRANSFER LEARNING (UNFROZEN)")
    print(f"{'='*60}")
    
    # Prepare data loaders
    train_dataset = TensorDataset(features_train.unsqueeze(1), y_train)
    val_dataset = TensorDataset(features_val.unsqueeze(1), y_val)
    test_dataset = TensorDataset(features_test.unsqueeze(1), y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    model = EEGConvNet(config.input_channels, config.input_height, config.input_width)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained weights (FIXED)
    loaded_state_dict = torch.load(pretrained_path, map_location=device, weights_only=False)
    
    # Remove 'module.' prefix if present (from DataParallel)
    if any(k.startswith('module.') for k in loaded_state_dict.keys()):
        loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
    
    # Load pretrained weights
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in loaded_state_dict.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # Replace head
    model.fc3 = NewHead(64)
    
    # Multi-GPU setup
    if config.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Train model
    model = train_model(model, train_loader, val_loader, config, model_suffix="_exp2_unfrozen")
    
    # Evaluate
    mae, rmse, mape = evaluate_model(model, test_loader, config, split_info, suffix=" - EXP2: TRANSFER (UNFROZEN)")
    
    return mae, rmse, mape
    
def run_experiment_3_transfer_frozen(features_train, features_val, features_test,
                                      y_train, y_val, y_test, config, split_info, pretrained_path):
    """Experiment 3: Transfer learning with frozen layers."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 3: TRANSFER LEARNING (FROZEN)")
    print(f"{'='*60}")
    
    # Prepare data loaders
    train_dataset = TensorDataset(features_train.unsqueeze(1), y_train)
    val_dataset = TensorDataset(features_val.unsqueeze(1), y_val)
    test_dataset = TensorDataset(features_test.unsqueeze(1), y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    model = EEGConvNet(config.input_channels, config.input_height, config.input_width)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained weights (FIXED)
    loaded_state_dict = torch.load(pretrained_path, map_location=device, weights_only=False)
    
    # Remove 'module.' prefix if present (from DataParallel)
    if any(k.startswith('module.') for k in loaded_state_dict.keys()):
        loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
    
    # Load pretrained weights
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in loaded_state_dict.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # Freeze all layers except the new head
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace head and unfreeze it
    model.fc3 = NewHead(64)
    for param in model.fc3.parameters():
        param.requires_grad = True
    
    # Multi-GPU setup
    if config.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Train model
    model = train_model(model, train_loader, val_loader, config, model_suffix="_exp3_frozen")
    
    # Evaluate
    mae, rmse, mape = evaluate_model(model, test_loader, config, split_info, suffix=" - EXP3: TRANSFER (FROZEN)")
    
    return mae, rmse, mape
    
def run_experiment_4_reve_baseline(features_train, features_val, features_test,
                                    y_train, y_val, y_test, config, split_info):
    """Experiment 4: REVE features + simple MLP regressor (baseline)."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 4: REVE BASELINE (MLP Regressor)")
    print(f"{'='*60}")
    
    # Fix the shape of features - ensure they are 2D (samples, features)
    print(f"Original shapes - Train: {features_train.shape}, Val: {features_val.shape}, Test: {features_test.shape}")
    
    # If features are 3D, reshape to 2D
    if len(features_train.shape) == 3:
        features_train = features_train.view(features_train.shape[0], -1)
        features_val = features_val.view(features_val.shape[0], -1)
        features_test = features_test.view(features_test.shape[0], -1)
        print(f"Reshaped to - Train: {features_train.shape}, Val: {features_val.shape}, Test: {features_test.shape}")
    
    # Initialize model
    model = REVERegressor(input_dim=features_train.shape[1])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Train model
    model = train_reve_regressor(model, features_train, y_train, features_val, y_val, 
                                  config, model_suffix="_exp4_reve")
    
    # Evaluate
    mae, rmse, mape = evaluate_reve_regressor(model, features_test, y_test, config, split_info, suffix=" - EXP4: REVE BASELINE")
    
    return mae, rmse, mape

# ============================================
# CONFIGURATION
# ============================================
class Config:
    # Data parameters
    dataset_path = "../Comprehension/Derco/"
    electrode_names = ['F3', 'F7', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8']
    sample_rate = 200
    
    # REVE model parameters
    hf_token = "[hugging_face_token]"
    reve_model = "brain-bzh/reve-base"
    reve_batch_size = 512
    
    # Training parameters
    batch_size = 1024
    learning_rate = 0.00001
    num_epochs = 500
    patience = 10
    eval_tolerance = 10
    
    # Participant split parameters
    n_val_participants = 2
    n_test_participants = 5
    random_seed = 42
    
    # Model architecture
    input_channels = 1
    input_height = 12  # Number of electrodes
    input_width = 512  # REVE feature dimension
    
    # Device
    use_multi_gpu = True
    
# ============================================
# MAIN PIPELINE
# ============================================
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Create config
config = Config()

# Print configuration
print("=" * 60)
print("CONFIGURATION")
print("=" * 60)
print(f"Dataset path: {config.dataset_path}")
print(f"Electrodes: {len(config.electrode_names)} channels")
print(f"Sample rate: {config.sample_rate} Hz")
print(f"REVE model: {config.reve_model}")
print(f"Batch size: {config.batch_size}")
print(f"Learning rate: {config.learning_rate}")
print(f"Max epochs: {config.num_epochs}")
print(f"Early stopping patience: {config.patience}")
print(f"Validation participants: {config.n_val_participants}")
print(f"Test participants: {config.n_test_participants}")
print(f"Random seed: {config.random_seed}")
print("=" * 60)

# Set random seeds
torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)

# Step 1: Load data with participant info
print("\n" + "="*60)
print("STEP 1: Loading EEG data with participant information")
print("="*60)
X, y, subjects = load_dataset_with_participants(
    config.dataset_path, config.electrode_names, config.sample_rate
)

# Step 2: Participant-based split
print("\n" + "="*60)
print("STEP 2: Participant-based data splitting")
print("="*60)
X_train, X_val, X_test, y_train, y_val, y_test, subjects_train, subjects_val, subjects_test, split_info = participant_based_split(
    X, y, subjects,
    n_val_participants=config.n_val_participants,
    n_test_participants=config.n_test_participants,
    random_state=config.random_seed
)

# Step 3: Extract REVE features
print("\n" + "="*60)
print("STEP 3: Extracting REVE features")
print("="*60)
features_train, features_val, features_test = extract_reve_features_for_splits(
    X_train, X_val, X_test, config.electrode_names, config
)

# Convert labels to tensors
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Step 4: Run Experiment 1 - CNN from scratch
mae_exp1, rmse_exp1, mape_exp1 = run_experiment_1_scratch(
    features_train, features_val, features_test,
    y_train, y_val, y_test, config, split_info
)

# Save the scratch model as pretrained model for transfer learning
pretrained_path = "best_model_exp1_scratch.pth"

# Step 5: Run Experiment 2 - Transfer learning (unfrozen)
mae_exp2, rmse_exp2, mape_exp2 = run_experiment_2_transfer_unfrozen(
    features_train, features_val, features_test,
    y_train, y_val, y_test, config, split_info, pretrained_path
)

# Step 6: Run Experiment 3 - Transfer learning (frozen)
mae_exp3, rmse_exp3, mape_exp3 = run_experiment_3_transfer_frozen(
    features_train, features_val, features_test,
    y_train, y_val, y_test, config, split_info, pretrained_path
)

# Step 7: Run Experiment 4 - REVE baseline (MLP regressor)
mae_exp4, rmse_exp4, mape_exp4 = run_experiment_4_reve_baseline(
    features_train, features_val, features_test,
    y_train, y_val, y_test, config, split_info
)

# Step 8: Summary of all experiments
print("\n" + "="*60)
print("ALL EXPERIMENTS SUMMARY")
print("="*60)
print(f"{'Experiment':<35} {'MAE':<10} {'RMSE':<10} {'MAPE (%)':<10}")
print("-" * 65)
print(f"{'Exp1: CNN from Scratch':<35} {mae_exp1:<10.4f} {rmse_exp1:<10.4f} {mape_exp1:<10.2f}")
print(f"{'Exp2: Transfer Learning (Unfrozen)':<35} {mae_exp2:<10.4f} {rmse_exp2:<10.4f} {mape_exp2:<10.2f}")
print(f"{'Exp3: Transfer Learning (Frozen)':<35} {mae_exp3:<10.4f} {rmse_exp3:<10.4f} {mape_exp3:<10.2f}")
print(f"{'Exp4: REVE Baseline (MLP)':<35} {mae_exp4:<10.4f} {rmse_exp4:<10.4f} {mape_exp4:<10.2f}")
print("="*60)

# Determine best experiment
results_dict = {
    'CNN from Scratch': mae_exp1,
    'Transfer Learning (Unfrozen)': mae_exp2,
    'Transfer Learning (Frozen)': mae_exp3,
    'REVE Baseline (MLP)': mae_exp4
}
best_exp = min(results_dict, key=results_dict.get)
print(f"\n🏆 Best performing experiment: {best_exp} (MAE: {results_dict[best_exp]:.4f})")
