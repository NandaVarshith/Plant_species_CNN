import os
import shutil
import random

SOURCE_DIR = r"C:\Users\HP\OneDrive\Desktop\NN\leafsnap-dataset-30subset\dataset\images\lab\Auto_cropped"
TRAIN_DIR  = r"C:\Users\HP\OneDrive\Desktop\NN\dataset\train"
TEST_DIR   = r"C:\Users\HP\OneDrive\Desktop\NN\dataset\test"

SPLIT_RATIO = 0.8

for species in os.listdir(SOURCE_DIR):
    species_dir = os.path.join(SOURCE_DIR, species)

    if not os.path.isdir(species_dir):
        continue

    # collect images from all subfolders safely
    images = []
    for root, dirs, files in os.walk(species_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(root, file))

    if len(images) == 0:
        continue

    random.shuffle(images)
    split_index = int(len(images) * SPLIT_RATIO)

    train_images = images[:split_index]
    test_images = images[split_index:]

    os.makedirs(os.path.join(TRAIN_DIR, species), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, species), exist_ok=True)

    for img_path in train_images:
        shutil.copy(img_path, os.path.join(TRAIN_DIR, species, os.path.basename(img_path)))

    for img_path in test_images:
        shutil.copy(img_path, os.path.join(TEST_DIR, species, os.path.basename(img_path)))

print("✅ Dataset split completed successfully without permission errors!")
