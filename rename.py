import os

# === CHANGE THESE ===
train_dir = "MFCN_SMC/model_input/input_Catheter_Whole_RANZCR/test/PICC"
mask_dir = "MFCN_SMC/model_input/mask_Catheter_Whole_RANZCR/test/PICC"

# === Get all files ===
image_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.jpg')])

# === Sanity check ===
assert len(image_files) == len(mask_files), "Number of images and masks does not match!"

# === Rename images and masks ===
for idx, (img, msk) in enumerate(zip(image_files, mask_files), 1):
    new_name = f"case{idx}.jpg"
    new_mask = f"case{idx}.jpg"

    os.rename(os.path.join(train_dir, img), os.path.join(train_dir, new_name))
    os.rename(os.path.join(mask_dir, msk), os.path.join(mask_dir, new_mask))

    print(f"Renamed: {img} -> {new_name} | {msk} -> {new_mask}")

print("All files renamed successfully.")
