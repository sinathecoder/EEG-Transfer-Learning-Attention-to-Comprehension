#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne

from sklearn.model_selection import train_test_split



# ============================================
# Utils
# ============================================

def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

# ============================================
# Dataset
# ============================================

def load_dataset(dataset_path):
    X, y = [], []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".fif"):
                file_path = os.path.join(root, file)

                epochs = mne.read_epochs(file_path, preload=True, verbose=False)\
                    .copy().resample(128, npad='auto')\
                    .pick(['F3','F7','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8'])

                data = epochs.get_data()
                labels = epochs.metadata['p_cloze'].values

                for i in range(len(data)):
                    if not np.isnan(labels[i]):
                        X.append(data[i])
                        y.append(labels[i])

    X = np.stack(X)
    y = np.array(y) * 100  # scale

    return X, y


def preprocess_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    mean, std = X_train.mean(), X_train.std()

    X_train = (X_train - mean) / std
    X_val   = (X_val - mean) / std
    X_test  = (X_test - mean) / std

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

# ============================================
# Model
# ============================================

class EEGConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 32, kernel_size=10, padding=2)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=10, padding=2)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=10, padding=2)
        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = x.squeeze(1)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class NewHead(nn.Module):
    def __init__(self, in_features=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ============================================
# Training
# ============================================

def train_model(model, X_train, y_train, X_val, y_val, params):
    device = params["device"]
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.L1Loss()

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(params["epochs"]):
        model.train()
        perm = torch.randperm(X_train.size(0))
        total_loss = 0

        for i in range(0, X_train.size(0), params["batch_size"]):
            idx = perm[i:i+params["batch_size"]]

            x = X_train[idx].to(device)
            y = y_train[idx].view(-1,1).to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for i in range(0, X_val.size(0), params["batch_size"]):
                x = X_val[i:i+params["batch_size"]].to(device)
                y = y_val[i:i+params["batch_size"]].view(-1,1).to(device)

                out = model(x)
                val_loss += criterion(out, y).item()

        print(f"Epoch {epoch} | Train {total_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= params["patience"]:
            print("Early stopping")
            break

    return model


# ============================================
# Evaluation
# ============================================

def evaluate(model, X_test, y_test, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for i in range(0, X_test.size(0), 64):
            x = X_test[i:i+64].to(device)
            y = y_test[i:i+64]

            out = model(x).squeeze()
            preds.extend(out.cpu().numpy())
            labels.extend(y.numpy())

    preds = np.array(preds)
    labels = np.array(labels)

    mae = np.mean(np.abs(preds - labels))
    print(f"MAE: {mae:.4f}")

# ============================================
# Experiments
# ============================================

def run_experiment(mode, pretrained_path, data, params):
    X_train, X_val, X_test, y_train, y_val, y_test = data

    print(f"\n=== Running: {mode} ===")

    model = EEGConvNet()

    # -----------------------------
    # 1. Train from scratch
    # -----------------------------
    if mode == "scratch":
        model.apply(reset_weights)

    # -----------------------------
    # 2. Transfer (unfrozen)
    # -----------------------------
    elif mode == "transfer_unfrozen":
    
        # Load pretrained model
        loaded_model = torch.load(pretrained_path, weights_only=False)
        if isinstance(loaded_model, torch.nn.DataParallel):
            loaded_model = loaded_model.module
        
        # Initialize current model (same architecture as checkpoint)
        model = EEGConvNet()  # or your base model
        
        # Load pretrained weights except fc2
        pretrained_dict = {k: v for k, v in loaded_model.state_dict().items() if "fc2" not in k}
        model.load_state_dict(pretrained_dict, strict=False)
        
        # Replace head
        in_features = model.fc2.in_features
        model.fc2 = NewHead()

    # -----------------------------
    # 3. Transfer (frozen)
    # -----------------------------
    elif mode == "transfer_frozen":
        # Load pretrained model
        loaded_model = torch.load(pretrained_path, weights_only=False)
        if isinstance(loaded_model, torch.nn.DataParallel):
            loaded_model = loaded_model.module
        
        # Initialize current model (same architecture as checkpoint)
        model = EEGConvNet()  # or your base model
        
        # Load pretrained weights except fc2
        pretrained_dict = {k: v for k, v in loaded_model.state_dict().items() if "fc2" not in k}
        model.load_state_dict(pretrained_dict, strict=False)

        for param in model.parameters():
            param.requires_grad = False

        model.fc2 = NewHead()

        # ensure head is trainable
        for param in model.fc2.parameters():
            param.requires_grad = True

    model = train_model(model, X_train, y_train, X_val, y_val, params)
    evaluate(model, X_test, y_test, params["device"])


# ============================================
# MAIN
# ============================================


import warnings
warnings.filterwarnings("ignore")


params = {
    "dataset_path": "../Comprehension/Derco/",
    "batch_size": 1024*4,
    "epochs": 1000,
    "lr": 0.0001,
    "patience": 5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "pretrained_path": "phyaat_exploration.pth"
}

print("Loading dataset...")
X, y = load_dataset(params["dataset_path"])
data = preprocess_data(X, y)

# Run all experiments
run_experiment("scratch", params["pretrained_path"], data, params)
run_experiment("transfer_unfrozen", params["pretrained_path"], data, params)
run_experiment("transfer_frozen", params["pretrained_path"], data, params)


