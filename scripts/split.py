import os
import random
import shutil

# Original dataset location
DATA_DIR = "maxar"
IMAGE_DIR = os.path.join(DATA_DIR, "images")  # Folder containing .tif images
LABEL_DIR = os.path.join(DATA_DIR, "labels")  # Folder containing .txt labels

# Train-test split ratio
TRAIN_RATIO = 0.8

# New folder where split data will be stored
OUTPUT_DIR = "data"
TRAIN_IMAGE_DIR = os.path.join(OUTPUT_DIR, "train", "images")
TRAIN_LABEL_DIR = os.path.join(OUTPUT_DIR, "train", "labels")
TEST_IMAGE_DIR = os.path.join(OUTPUT_DIR, "test", "images")
TEST_LABEL_DIR = os.path.join(OUTPUT_DIR, "test", "labels")

# Create train/test folders in the new location
for folder in [TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, TEST_IMAGE_DIR, TEST_LABEL_DIR]:
    os.makedirs(folder, exist_ok=True)

# Get all images and filter only those with corresponding labels
all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".tif")]
valid_images = []

for img_file in all_images:
    label_file = os.path.splitext(img_file)[0] + ".txt"  # Corresponding label file
    if os.path.exists(os.path.join(LABEL_DIR, label_file)):  # Only keep valid pairs
        valid_images.append(img_file)

# Debugging: Check detected files
if not valid_images:
    print("❌ No valid image-label pairs found! Check your dataset folder.")
    exit()

# Shuffle and split dataset
random.shuffle(valid_images)
split_idx = int(TRAIN_RATIO * len(valid_images))
train_images = valid_images[:split_idx]
test_images = valid_images[split_idx:]

# Function to copy files
def copy_files(image_list, src_img_dir, src_lbl_dir, dest_img_dir, dest_lbl_dir):
    for img_file in image_list:
        img_path = os.path.join(src_img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"  # Label name
        label_path = os.path.join(src_lbl_dir, label_file)

        # Copy image and label
        shutil.copy(img_path, os.path.join(dest_img_dir, img_file))
        shutil.copy(label_path, os.path.join(dest_lbl_dir, label_file))

# Copy valid training and testing files to `split_data/`
copy_files(train_images, IMAGE_DIR, LABEL_DIR, TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)
copy_files(test_images, IMAGE_DIR, LABEL_DIR, TEST_IMAGE_DIR, TEST_LABEL_DIR)

print(f"✅ Split completed: {len(train_images)} training images & {len(test_images)} testing images.")
print(f"📂 Only valid (image, label) pairs are saved in 'split_data/'.")
