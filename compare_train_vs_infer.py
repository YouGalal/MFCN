import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
import random
from glob import glob

# Import your model and header
import model
import header

# === Directory settings ===
img_dir  = './MFCN_SMC/output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/First_output/random_crop_train/data/input_Catheter__Whole_RANZCR/PICC'
mask_dir = './MFCN_SMC/output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/First_output/random_crop_train/data/mask_Catheter__Whole_RANZCR/PICC'

def get_random_case_id():
    image_files = sorted(glob(os.path.join(img_dir, '*.jpg')))
    random_file = random.choice(image_files)
    case_id = os.path.basename(random_file).replace('.jpg', '')
    return case_id

def preprocess_image(path):
    img = np.array(Image.open(path).convert('L'))
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, 1))  # shape [1, 1, H, W]
    return torch.tensor(img, dtype=torch.float32)

def load_gt_mask(path):
    mask = np.array(Image.open(path).convert('L'))
    mask = (mask > 127).astype(np.float32)
    return mask

def load_model(eval_mode=True):
    net = model.FCDenseNet(header.num_channel, header.num_masks)
    ckpt = torch.load(os.path.join(header.dir_checkpoint, 'FCDenseNet_Stage2_epoch9.pth'), map_location='cpu')
    net.load_state_dict(ckpt['model_state_dict'])

    if eval_mode:
        net.train()
    else:
        net.train()
    return net

def predict(net, img_tensor):
    with torch.no_grad():
        out = net(img_tensor)
        out = torch.sigmoid(out)
        return out.squeeze().numpy()

def show_results(case_id, train_pred, infer_pred, original, gt_mask):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(original.squeeze(), cmap='gray')
    axs[0].set_title("Input Image")

    axs[1].imshow(gt_mask, cmap='gray')
    axs[1].set_title("Ground Truth")

    axs[2].imshow((train_pred > 0.5).astype(np.uint8), cmap='gray')
    axs[2].set_title("Train Mode Prediction")

    axs[3].imshow((infer_pred > 0.5).astype(np.uint8), cmap='gray')
    axs[3].set_title("Inference Mode Prediction")

    for ax in axs:
        ax.axis('off')
    plt.suptitle(f"Case ID: {case_id}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def main():
    case_id = get_random_case_id()
    print(f"\n>>> Selected Random Case: {case_id}")

    img_path = os.path.join(img_dir, case_id + '.jpg')
    mask_path = os.path.join(mask_dir, case_id + '.jpg')

    img_tensor = preprocess_image(img_path)
    original = img_tensor.numpy()
    gt_mask = load_gt_mask(mask_path)

    net_train = load_model(eval_mode=False)
    net_infer = load_model(eval_mode=True)

    train_pred = predict(net_train, img_tensor)
    infer_pred = predict(net_infer, img_tensor)

    show_results(case_id, train_pred, infer_pred, original, gt_mask)

if __name__ == "__main__":
    main()
