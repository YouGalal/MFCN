import os
import torch
import matplotlib.pyplot as plt

checkpoint_dir = "./MFCN_SMC/checkpoint/"
train_losses = []
val_losses = []
epochs = []

files = sorted([
    f for f in os.listdir(checkpoint_dir)
    if f.startswith("FCDenseNet_Stage3_epoch") and f.endswith(".pth")
], key=lambda x: int(x.split("epoch")[1].split(".")[0]))

for f in files:
    path = os.path.join(checkpoint_dir, f)
    try:
        ckpt = torch.load(path, map_location='cpu')
        train_losses.append(ckpt['train_loss'])
        val_losses.append(ckpt['val_loss'])
        epochs.append(ckpt['epoch'])
    except Exception as e:
        print(f"Skipping {f}: {e}")

if epochs:
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Stage 3 Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("stage3_loss_plot.png")
    plt.show()
else:
    print("No valid checkpoints were loaded.")