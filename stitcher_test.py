import os
import numpy as np
from PIL import Image
from collections import defaultdict

# --------------------------
# CONFIG - ADJUST THIS!
# --------------------------
original_size = (3000, 3000)  # (width, height) -- adjust!
patch_folder   = "./MFCN_SMC/output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/First_output/random_crop/data/input_Catheter__Whole_RANZCR/PICC/"  # folder with your 2000 patches
output_dir    = "./stitched_check/"

os.makedirs(output_dir, exist_ok=True)

# === Group patches by case ===
cases = defaultdict(list)

for fname in os.listdir(patch_folder):
    if not fname.endswith(".png"):
        continue
    parts = fname.split("_")
    case_name = parts[0]
    cases[case_name].append(fname)

print(f"Found {len(cases)} cases: {list(cases.keys())}")

# === Process each case ===
for case_name, files in cases.items():
    canvas = np.zeros((original_size[1], original_size[0]), dtype=np.float32)
    counter = np.zeros((original_size[1], original_size[0]), dtype=np.float32)

    for fname in files:
        patch_path = os.path.join(patch_folder, fname)
        patch = np.array(Image.open(patch_path).convert('L'))

        parts = fname.replace('.png', '').split('_')
        x1, y1, x2, y2 = map(int, parts[2:6])

        # Intended slot size
        H = y2 - y1
        W = x2 - x1

        # Actual patch size
        ph, pw = patch.shape

        # Safe overlap: match patch, slot, canvas
        hh = min(H, ph, canvas.shape[0] - y1)
        ww = min(W, pw, canvas.shape[1] - x1)

        # Skip degenerate
        if hh <= 0 or ww <= 0:
            print(f"⚠️ Skipped patch {fname} due to invalid overlap.")
            continue

        canvas[y1:y1+hh, x1:x1+ww] += patch[:hh, :ww]
        counter[y1:y1+hh, x1:x1+ww] += 1

    counter[counter == 0] = 1
    stitched = (canvas / counter).astype(np.uint8)

    out_path = os.path.join(output_dir, f"{case_name}_stitched.png")
    Image.fromarray(stitched).save(out_path)
    print(f"✅ Saved: {out_path}")

print("🎉 Done stitching all cases.")