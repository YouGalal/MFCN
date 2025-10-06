import cv2
import numpy as np
import os

# Paths
xray_folder = "../model_input/input_Catheter_Whole_RANZCR/test/PICC"   # folder with X-ray images
pred_folder = "../output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/FCDenseNetthird_output/FCDenseNet/First_connected_component"     # folder with prediction masks
output_folder = "../output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/FCDenseNetthird_output/FCDenseNet/Overlay"  # output folder

# Create output folder if it doesn’t exist
os.makedirs(output_folder, exist_ok=True)

# Loop over all X-ray images
for filename in os.listdir(xray_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        xray_path = os.path.join(xray_folder, filename)
        pred_path = os.path.join(pred_folder, filename)  # assumes same name

        if not os.path.exists(pred_path):
            print(f"Skipping {filename} (no matching prediction found)")
            continue

        # Load images
        xray = cv2.imread(xray_path)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if xray is None or pred is None:
            print(f"Error loading {filename}, skipping...")
            continue

        # Resize prediction to match X-ray
        pred = cv2.resize(pred, (xray.shape[1], xray.shape[0]))

        # Threshold mask
        _, mask = cv2.threshold(pred, 1, 255, cv2.THRESH_BINARY)

        # Create red overlay
        red_overlay = np.zeros_like(xray, dtype=np.uint8)
        red_overlay[mask > 0] = (0, 0, 255)

        # Blend
        alpha = 0.4
        result = cv2.addWeighted(red_overlay, alpha, xray, 1 - alpha, 0)

        # Save result
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, result)

        print(f"Saved overlay: {out_path}")
