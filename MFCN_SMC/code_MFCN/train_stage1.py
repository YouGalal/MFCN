# train_stage1.py

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
    print("\nStarting Stage 1 Training (Whole-Image Segmentation)\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net = header.First_net_model[0]  # e.g., FC-DenseNet
    net = net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    print("Model loaded:", header.First_net_string[0])

    # Dataset & DataLoader
    train_count = len(os.listdir(header.dir_train_path))
    val_count = len(os.listdir(header.dir_valid_path))

    print(f"Found {train_count} training images.")
    print(f"Found {val_count} validation images.")

    train_indices = mydataset.shuffle_dataset(train_count)
    valid_indices = mydataset.shuffle_dataset(val_count)

    train_set = mydataset.MyTrainDataset(header.dir_train_path, train_indices)
    val_set   = mydataset.MyTrainDataset(header.dir_valid_path, valid_indices)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=header.num_worker)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=header.num_worker)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=header.learning_rate)

    for epoch in range(header.epoch_max):
        print(f"--- Starting Epoch {epoch+1}/{header.epoch_max} ---")
        net.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):

            try:
                inputs = batch['input'].float().to(device)
                labels = batch['masks'].long().to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 50 == 0:
                    print(f"[Epoch {epoch+1}/{header.epoch_max}] Batch {i} Loss: {loss.item():.8f}")

            except Exception as e:
                print(f" ERROR in Batch {i+1}: {e}")
                raise


        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{header.epoch_max}] Training Loss: {avg_loss:.8f}")

        # Validate
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
        print(f"Epoch [{epoch+1}/{header.epoch_max}] Validation Loss: {avg_val_loss:.8f}")

        # Save
        save_path = os.path.join(header.dir_checkpoint, f"{header.First_net_string[0]}_epoch{epoch+1}.pth")
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
