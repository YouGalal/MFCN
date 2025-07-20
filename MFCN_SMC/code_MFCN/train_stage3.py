import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from glob import glob
import random

import header
import model

# VMFLG Fragment Generator
def apply_vmflg(mask, num_fragments=3):
    mask_np = np.array(mask)
    frag = mask_np.copy()
    coords = np.column_stack(np.where(mask_np > 0))
    if len(coords) == 0:
        return Image.fromarray(frag)
    for _ in range(num_fragments):
        yx = coords[random.randint(0, len(coords) - 1)]
        x, y = yx[1], yx[0]
        r = random.randint(10, 50)
        cv2.circle(frag, (x, y), r, 0, -1)
    return Image.fromarray(frag)

# Dataset Class
class VMFLGMaskDataset(Dataset):
    def __init__(self, mask_dir, apply_vmflg=True):
        self.mask_paths = sorted(glob(os.path.join(mask_dir, '*.jpg')))
        self.apply_vmflg = apply_vmflg

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        gt = np.array(Image.open(self.mask_paths[idx]).convert('L'))  # [H, W]
        gt = (gt > 127).astype(np.float32)

        if self.apply_vmflg:
            frag = apply_vmflg(Image.fromarray((gt * 255).astype(np.uint8)), random.randint(1, 5))
            frag = np.array(frag).astype(np.float32) / 255.0
        else:
            frag = gt.copy()

        frag = cv2.resize(frag, (512,512), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (512,512), interpolation=cv2.INTER_NEAREST)

        frag = np.expand_dims(frag, axis=0)  # [1, H, W]
        gt = np.expand_dims(gt, axis=0)

        return {
            'input': torch.tensor(frag, dtype=torch.float32),
            'masks': torch.tensor(gt, dtype=torch.float32)
        }

# Main Training Loop
def main():
    print("\n=== Stage 3 VMFLG Training ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_mask_dir = '../model_input/mask_Catheter_Whole_RANZCR/train'
    val_mask_dir   = '../model_input/mask_Catheter_Whole_RANZCR/validation'

    train_dataset = VMFLGMaskDataset(train_mask_dir, apply_vmflg=True)
    val_dataset   = VMFLGMaskDataset(val_mask_dir, apply_vmflg=True)

    train_loader = DataLoader(train_dataset, batch_size=header.num_batch_train, shuffle=True, num_workers=header.num_worker)
    val_loader   = DataLoader(val_dataset, batch_size=header.num_batch_train, shuffle=False, num_workers=header.num_worker)

    print(f"\n>>> Loaded training masks: {len(train_dataset)}")
    print(f">>> Loaded validation masks: {len(val_dataset)}\n")

    net = model.FCDenseNet(header.num_channel, header.num_masks).to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=header.learning_rate)

    for epoch in range(header.epoch_max):
        net.train()
        train_loss = 0.0

        for i, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            labels = batch['masks'].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Visualize occasionally
            if i % 10 == 0:
                import matplotlib.pyplot as plt
                pred_mask = torch.sigmoid(outputs[0]).detach().cpu().squeeze().numpy()
                pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)
                input_img = inputs[0].detach().cpu().squeeze().numpy()
                target_mask = labels[0].detach().cpu().squeeze().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(input_img, cmap='gray')
                axes[0].set_title('Fragmented Input')
                axes[1].imshow(target_mask, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[2].imshow(pred_mask_bin, cmap='gray')
                axes[2].set_title('Predicted Mask')
                for ax in axes:
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(f'plots3/vis_epoch{epoch+1}_batch{i}.png')
                plt.close()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}] Batch {i} Train Loss: {loss.item():.6f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{header.epoch_max}] Train Loss: {avg_train_loss:.6f}")

        # Validation
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                labels = batch['masks'].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{header.epoch_max}] Val Loss: {avg_val_loss:.6f}")

        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        ckpt_path = os.path.join(header.dir_checkpoint, f'FCDenseNet_Stage3_epoch{epoch+1}.pth')
        torch.save(ckpt, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    print("\n=== Done. Stage 3 training complete. ===")


if __name__ == "__main__":
    main()