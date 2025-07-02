# train_stage2.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import header
import mydataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def main():
    print("\n========== Starting Stage 2 Training (Patch-wise Refinement) ==========\n")

    # 🧩 Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 🧩 Model
    net = header.First_net_model[0]  # Same architecture, fresh weights unless you want to load stage1 weights
    net = net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    print(f"Stage 2 Model: {header.First_net_string[0]}")

    # 🧩 Patch dataset paths (must match your crop script output!)
    patch_input_dir = "../output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/First_output/random_crop/data/input_Catheter__Whole_RANZCR/PICC" 
    patch_mask_dir  = "../output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/First_output/random_crop/data/mask_Catheter__Whole_RANZCR/PICC"

    if not os.path.exists(patch_input_dir):
        raise FileNotFoundError(f"Input patch dir not found: {patch_input_dir}")
    if not os.path.exists(patch_mask_dir):
        raise FileNotFoundError(f"Mask patch dir not found: {patch_mask_dir}")

    input_patches = sorted([f for f in os.listdir(patch_input_dir) if f.endswith(".jpg")])
    mask_patches  = sorted([f for f in os.listdir(patch_mask_dir) if f.endswith(".jpg")])

    if len(input_patches) == 0 or len(mask_patches) == 0:
        raise RuntimeError("No patches found! Make sure you ran Stage 1 + connected_component_crop correctly.")

    print(f"Found {len(input_patches)} input patches & {len(mask_patches)} mask patches")

    # Sanity check: assume both lists have matching names (by your pipeline)
    assert set([x.split("_")[0] for x in input_patches]) == set([x.split("_")[0] for x in mask_patches]), \
        "Mismatch between input and mask patch names!"

    # 🧩 Shuffle & split
    indices = mydataset.shuffle_dataset(len(input_patches))
    train_indices = indices[:int(0.8 * len(indices))]
    val_indices   = indices[int(0.8 * len(indices)):]

    train_set = mydataset.MyTrainDataset(patch_input_dir, train_indices)
    val_set   = mydataset.MyTrainDataset(patch_input_dir, val_indices)  # Same input dir — MyTrainDataset auto-finds masks

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=header.num_worker)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=header.num_worker)

    print(f"Training patches: {len(train_set)} | Validation patches: {len(val_set)}")

    # 🧩 Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=header.learning_rate)

    for epoch in range(header.epoch_max):
        net.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            inputs = batch['input'].float().to(device)
            labels = batch['masks'].long().to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 0:
                print(f"[Epoch {epoch+1}/{header.epoch_max}] Batch {i} Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{header.epoch_max}] Training Loss: {avg_loss:.4f}")

        # ✅ Validation
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].float().to(device)
                labels = batch['masks'].long().to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{header.epoch_max}] Validation Loss: {avg_val_loss:.4f}")

        # ✅ Save checkpoint
        save_path = os.path.join(header.dir_checkpoint, f"{header.First_net_string[0]}_Stage2_epoch{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'val_loss': avg_val_loss
        }, save_path)
        print(f"Saved checkpoint: {save_path}")

if __name__ == "__main__":
    main()
