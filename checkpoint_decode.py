import torch
import os

# 👉 Replace this with your checkpoint folder
checkpoint_dir = "C:/Users/yoya/dj3s/bsc/pytorch/MFCN/MFCN_SMC/checkpoint"

# List all checkpoint files
files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])

for file in files:
    checkpoint = torch.load(os.path.join(checkpoint_dir, file), map_location="cpu")
    epoch = checkpoint.get("epoch")
    train_loss = checkpoint.get("train_loss")
    val_loss = checkpoint.get("val_loss")
    print(f"{file}: Epoch {epoch} - Train Loss: {train_loss:.8f} - Val Loss: {val_loss:.8f}")
