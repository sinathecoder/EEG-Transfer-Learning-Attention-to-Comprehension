#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from mne.preprocessing import ICA
from mne_icalabel import label_components
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# FUNCTION DEFINITIONS
# ============================================================================

def load_csv_files(folder_path):
    """Load all CSV files matching pattern from folder and subfolders."""
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith("Signals.csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def create_sequences(df, window_size, overlap, ch_names, label_col="Label_S"):
    """Create sliding window sequences from EEG data."""
    step = int(window_size * (1 - overlap))
    sequences, labels = [], []
    
    feature_cols = [col for col in df.columns if col in ch_names]
    
    i = 0
    n = len(df)
    
    while i + window_size <= n:
        window = df.iloc[i:i+window_size]
        
        if window[label_col].nunique() == 1:
            seq = window[feature_cols].values
            label = window[label_col].iloc[0]
            sequences.append(seq)
            labels.append(label)
            i += step
        else:
            label_series = window[label_col].values
            change_idx = np.where(label_series != label_series[0])[0]
            i += change_idx[0] if len(change_idx) > 0 else step
    
    return np.array(sequences), np.array(labels)

def preprocess_eeg(X, y, sfreq, ch_names, random_seed):
    """Apply bandpass filter, re-referencing, ICA artifact removal, and autoreject."""
    
    # Convert to MNE format: (n_epochs, n_channels, n_times)
    X_mne = X.transpose(0, 2, 1)
    metadata = pd.DataFrame({'label': y})
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    epochs = mne.EpochsArray(X_mne, info, metadata=metadata)
    epochs.set_montage(mne.channels.make_standard_montage('standard_1020'))
    
    # Filtering
    epochs.filter(l_freq=0.1, h_freq=45, method='fir', fir_window='hamming')
    
    # Re-referencing
    epochs.set_eeg_reference('average')
    
    # Outlier removal (FASTER-like)
    data = epochs.get_data()
    amp = data.max(axis=2) - data.min(axis=2)
    var = data.var(axis=2)
    
    z_amp, z_var = zscore(amp, axis=0), zscore(var, axis=0)
    bad_epochs = (np.abs(z_amp) > 3).any(axis=1) | (np.abs(z_var) > 3).any(axis=1)
    epochs = epochs.drop(np.where(bad_epochs)[0])
    
    # ICA for artifact removal
    ica = ICA(method='picard', n_components=0.99, random_state=random_seed)
    ica.fit(epochs)
    
    labels = label_components(epochs, ica, method='iclabel')
    bad_idx = [i for i, label in enumerate(labels['labels']) 
               if label in ['eye', 'muscle', 'heart', 'line_noise']]
    ica.exclude = bad_idx
    epochs_clean = ica.apply(epochs.copy())
    
    return epochs_clean.get_data(), y[epochs_clean.selection]

class EEGConvNet(nn.Module):
    """EEG Convolutional Neural Network for classification."""
    
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 32, kernel_size=10, padding=2)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=10, padding=2)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=10, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        x = x.squeeze(1)  # Remove channel dim
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train_model(model, X_train, y_train, X_val, y_val, device, 
                num_epochs, batch_size, patience, learning_rate):
    """Train model with early stopping."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        perm = torch.randperm(X_train.size(0))
        
        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i+batch_size]
            batch_x, batch_y = X_train[idx].to(device), y_train[idx].to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for i in range(0, X_val.size(0), batch_size):
                batch_x, batch_y = X_val[i:i+batch_size].to(device), y_val[i:i+batch_size].to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_acc = correct / total
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_eeg_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    return model

def evaluate_model(model, X_test, y_test, device, batch_size=64):
    """Evaluate model on test set."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for i in range(0, X_test.size(0), batch_size):
            batch_x, batch_y = X_test[i:i+batch_size].to(device), y_test[i:i+batch_size].to(device)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    return all_preds, all_labels

def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    """Plot confusion matrix with counts and percentages."""
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, 
                yticklabels=labels, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix (Counts)')
    
    # Percentages
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax2,
                cbar_kws={'label': 'Percentage (%)'})
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix (Percentages)')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN EXECUTION CODE
# ============================================================================


# ------------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------------

# Data parameters
DATA_FOLDER = '../Attention/phyaat/phyaat_dataset/Signals'
CH_NAMES = ['F3', 'F7', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8']
WINDOW_SIZE = 145
OVERLAP = 0.2
SAMPLE_RATE = 128
RANDOM_SEED = 42

# Training parameters
NUM_EPOCHS = 150
BATCH_SIZE = 1024
PATIENCE = 5
LEARNING_RATE = 0.0001
TEST_SPLIT_SIZE = 0.4
VALIDATION_SPLIT_SIZE = 0.5  # 0.5 of remaining (so 0.2 of total)

# Model parameters
NUM_CLASSES = 3

# ------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------

print("Loading CSV files...")
csv_files = load_csv_files(DATA_FOLDER)
all_X, all_y = [], []

for path in csv_files:
    data = pd.read_csv(path)
    X, y = create_sequences(data, WINDOW_SIZE, OVERLAP, CH_NAMES, label_col="Label_S")
    all_X.append(X)
    all_y.append(y)

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
print(f"Loaded {len(X)} samples with shape {X.shape}")

# ------------------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------------------

print("\nPreprocessing EEG data...")
X_clean, y_clean = preprocess_eeg(X, y, SAMPLE_RATE, CH_NAMES, RANDOM_SEED)
print(f"After preprocessing: {X_clean.shape}")

# ------------------------------------------------------------------------
# TRAIN/VAL/TEST SPLIT
# ------------------------------------------------------------------------

print("\nSplitting data...")
# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_clean)
print(f"Label mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X_clean, y_encoded, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=VALIDATION_SPLIT_SIZE, random_state=RANDOM_SEED
)

print(f"Train: {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")

# Normalize
mean, std = X_train.mean(), X_train.std()
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# ------------------------------------------------------------------------
# MODEL SETUP
# ------------------------------------------------------------------------

print("\nSetting up model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Multi-GPU support
model = EEGConvNet(num_classes=NUM_CLASSES)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

# ------------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------------

print("\nStarting training...")
model = train_model(
    model, X_train, y_train, X_val, y_val, device,
    NUM_EPOCHS, BATCH_SIZE, PATIENCE, LEARNING_RATE
)

# ------------------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------------------

print("\nEvaluating on test set...")
preds, labels = evaluate_model(model, X_test, y_test, device)
test_acc = accuracy_score(labels, preds)
print(f"Test Accuracy: {test_acc:.4f}")

# ------------------------------------------------------------------------
# RESULTS VISUALIZATION
# ------------------------------------------------------------------------

print("\nGenerating confusion matrix...")
cm = confusion_matrix(labels, preds)
plot_confusion_matrix(cm, labels=["high", "low", "medium"])

# Print class distribution
unique, counts = np.unique(y_clean, return_counts=True)
print(f"\nClass distribution in cleaned data: {dict(zip(unique, counts))}")

print("\nDone!")

