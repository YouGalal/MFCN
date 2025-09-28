import pandas as pd
import os

# Load Excel file
file_path = "MFCN_SMC/output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/FCDenseNetthird_output/FCDenseNetthird_image_rmse_jpg_mm_worst.xlsx" 
df = pd.read_excel(file_path)

file_name = os.path.basename(file_path)
print(f"\nResults for file: {file_name}\n")

# df = df[~df['image_name'].str.contains("case144|case77|case14|case12|case48", case=False, na=False)]

# Columns to analyze
columns = ["dice", "subin", "point_rmse", "point_rmse_mm"]

for col in columns:
    mean_val = df[col].mean()
    range_val = df[col].max() - df[col].min()
    print(f"{col} -> {mean_val:.4f} ± {range_val:.4f}") 