import os

# Define the target directory
folder = "./MFCN_SMC/model_input/mask_Catheter_Whole_RANZCR/train"

# List and sort the JPG files
files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
files.sort()

# Rename each file
for i, filename in enumerate(files, start=1):
    old_path = os.path.join(folder, filename)
    new_filename = f"case{i}.jpg"
    new_path = os.path.join(folder, new_filename)

    if os.path.exists(new_path):
        print(f"Skipping {new_filename} (already exists)")
        continue

    os.rename(old_path, new_path)
    print(f"Renamed: {filename} -> {new_filename}")

print("\nDone renaming files.")
