# train_stage2.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from glob import glob
import torch.nn.functional as F

import header
import model

class PatchDatasetFromFolder(Dataset):
    def __init__(self, image_folder, mask_folder):
        self.image_paths = sorted(glob(os.path.join(image_folder, '*.jpg')))
        self.mask_paths = sorted(glob(os.path.join(mask_folder, '*.jpg')))
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch in number of images and masks"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert('L'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))

        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
        img = img.astype(np.float32) / 255.0

        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)  # Shape: [1, H, W]

        img = np.expand_dims(img, axis=0)

        # print(f"[Debug] Input shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")

        return {
            'input': torch.tensor(img, dtype=torch.float32),
            'masks': torch.tensor(mask, dtype=torch.float32)
        }


def main():

    print("\n=== Stage 2 Patch-Wise Training (Using Cropped Patches) ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Load cropped patches ===
    train_cxr_dir = '../output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/First_output/random_crop_train/data/input_Catheter__Whole_RANZCR/PICC'
    train_mask_dir = '../output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/First_output/random_crop_train/data/mask_Catheter__Whole_RANZCR/PICC'

    val_cxr_dir = '../output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/First_output/random_crop_validation/data/input_Catheter__Whole_RANZCR/PICC'
    val_mask_dir = '../output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/First_output/random_crop_validation/data/mask_Catheter__Whole_RANZCR/PICC'

    train_dataset = PatchDatasetFromFolder(train_cxr_dir, train_mask_dir)
    val_dataset   = PatchDatasetFromFolder(val_cxr_dir, val_mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=header.num_batch_train, shuffle=True, num_workers=header.num_worker)
    val_loader   = DataLoader(val_dataset, batch_size=header.num_batch_train, shuffle=False, num_workers=header.num_worker)

    print(f"\n>>> Loaded training images: {len(train_dataset)}")
    print(f">>> Loaded validation images: {len(val_dataset)}\n")

    ### training ###

    net = model.FCDenseNet(header.num_channel, header.num_masks1).to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=header.learning_rate)

    for epoch in range(header.epoch_max):
        net.train()
        train_loss = 0.0

        print(">>> Starting training loop")
        for i, batch in enumerate(train_loader):
            inputs = batch['input'].float().to(device)
            labels = batch['masks'].float().to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            if i % 100 == 0:  # visualize every 100 batches
                import matplotlib.pyplot as plt
                import torchvision

                input_img = inputs[0].detach().cpu().squeeze().numpy()         # [H, W]
                target_mask = labels[0].detach().cpu().squeeze().numpy()       # [H, W]
                pred_mask = torch.sigmoid(outputs[0]).detach().cpu().squeeze().numpy()  # [H, W]
                pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(input_img, cmap='gray')
                axes[0].set_title('Input Image')
                axes[1].imshow(target_mask, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[2].imshow(pred_mask_bin, cmap='gray')
                axes[2].set_title('Predicted Mask')

                for ax in axes:
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(f'plots/vis_epoch{epoch+1}_batch{i}.png')
                plt.close()

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}] Batch {i} Train Loss: {loss.item():.6f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{header.epoch_max}] Train Loss: {avg_train_loss:.6f}")

        # Validation loop
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].float().to(device)
                labels = batch['masks'].float().to(device)

                outputs = net(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{header.epoch_max}] Val Loss: {avg_val_loss:.6f}")

        ckpt = {
            'epoch': epoch+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        ckpt_path = os.path.join(header.dir_checkpoint, f'FCDenseNet_Stage2_epoch{epoch+1}.pth')
        torch.save(ckpt, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    print("\n=== Done. Stage 2 trained with explicit validation ===")


if __name__ == "__main__":
    main()