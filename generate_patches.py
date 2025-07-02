# generate_patches.py

import os
import cv2
import numpy as np
from PIL import Image

PATCH_SIZE = 512
PATCHES_PER_IMAGE = 5  # How many patches to extract per image

# === PATHS ===
original_dir = "../model_input/input_Catheter_Whole_RANZCR/train"
predicted_mask_dir = "../output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/First_output/Whole"
gt_mask_dir = "../model_input/mask_Catheter_Whole_RANZCR/train"

patch_output_dir = "../model_input/patch_input/train"

os.makedirs(patch_output_dir, exist_ok=True)
os.makedirs(patch_output_dir + "/images", exist_ok=True)
os.makedirs(patch_output_dir + "/masks", exist_ok=True)

def extract_patches(image, mask, gt_mask, base_name):
    h, w = mask.shape
    mask_coords = np.argwhere(mask > 0)

    for i in range(PATCHES_PER_IMAGE):
        if len(mask_coords) == 0:
            break

        # Pick a random pixel where mask = 1
        y, x = mask_coords[np.random.choice(len(mask_coords))]

        # Define patch bounds
        top = max(y - PATCH_SIZE // 2, 0)
        left = max(x - PATCH_SIZE // 2, 0)
        bottom = min(top + PATCH_SIZE, h)
        right = min(left + PATCH_SIZE, w)

        # Adjust if at edge
        top = bottom - PATCH_SIZE
        left = right - PATCH_SIZE
        top = max(top, 0)
        left = max(left, 0)

        # Crop
        img_patch = image[top:bottom, left:right]
        mask_patch = gt_mask[top:bottom, left:right]

        # Save
        img_pil = Image.fromarray(img_patch)
        mask_pil = Image.fromarray(mask_patch)

        img_pil.save(f"{patch_output_dir}/images/{base_name}_patch{i}.png")
        mask_pil.save(f"{patch_output_dir}/masks/{base_name}_patch{i}.png")

        print(f"Saved: {base_name}_patch{i}.png")

def main():
    image_files = [f for f in os.listdir(original_dir) if f.endswith(".jpg")]

    for img_file in image_files:
        base_name = img_file.replace(".jpg", "")

        # Load original, predicted, and GT mask
        image = np.array(Image.open(os.path.join(original_dir, img_file)).convert("L"))
        predicted_mask = np.array(Image.open(os.path.join(predicted_mask_dir, f"{base_name}_mask.jpg")).convert("L"))
        gt_mask = np.array(Image.open(os.path.join(gt_mask_dir, img_file)).convert("L"))

        # Binarize predicted mask if needed
        predicted_mask = (predicted_mask > 127).astype(np.uint8)

        extract_patches(image, predicted_mask, gt_mask, base_name)

if __name__ == "__main__":
    main()
