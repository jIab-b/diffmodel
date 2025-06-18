import os
from datasets import Dataset

# Path to your .arrow file
arrow_path = "c:/Users/Beedf/.cache/huggingface/datasets/huggan___smithsonian_butterflies_subset/default/0.0.0/3cdedf844922ab40393d46d4c7f81c596e1c6d45/smithsonian_butterflies_subset-train.arrow"

# Load the dataset from the .arrow file
ds = Dataset.from_file(arrow_path)  # [4]

# Name of the column containing images
image_column = "image"  # Change this if your column is named differently

# Create output directory
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# Loop through the dataset and save images
for idx, item in enumerate(ds):
    img = item[image_column]
    # If the image is a PIL Image object, save directly
    if hasattr(img, "save"):
        img.save(os.path.join(output_dir, f"image_{idx:05d}.png"))
    # If the image is bytes, convert to PIL and save
    elif isinstance(img, bytes):
        from PIL import Image
        from io import BytesIO
        Image.open(BytesIO(img)).save(os.path.join(output_dir, f"image_{idx:05d}.png"))
    # If the image is a file path, copy or move as needed
    elif isinstance(img, str) and os.path.exists(img):
        from shutil import copyfile
        copyfile(img, os.path.join(output_dir, f"image_{idx:05d}.png"))
    else:
        print(f"Unknown image format at index {idx}")

print(f"Saved {len(ds)} images to '{output_dir}'")
