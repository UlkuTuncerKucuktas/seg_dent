import os
import json
import random
import shutil
from tqdm import tqdm

def convert_coco_to_yolo_segmentation(json_path, images_folder, output_folder, val_split=0.2):
    # Load the COCO JSON file
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Create directories for YOLO dataset
    os.makedirs(os.path.join(output_folder, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'labels', 'val'), exist_ok=True)

    # Dictionary to hold image dimensions
    image_dims = {img["id"]: (img["width"], img["height"], img["file_name"]) for img in coco_data["images"]}

    # Collect existing images and their annotations
    annotations = {}
    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        img_width, img_height, file_name = image_dims[image_id]

        if file_name not in annotations:
            annotations[file_name] = []

        # Segmentation
        segmentation = annotation["segmentation"]
        normalized_segmentation = []
        for seg in segmentation:
            normalized_seg = [(seg[i] / img_width if i % 2 == 0 else seg[i] / img_height) for i in range(len(seg))]
            normalized_segmentation.extend(normalized_seg)

        annotations[file_name].append([0] + normalized_segmentation)

    # Split the dataset into training and validation sets
    images = list(annotations.keys())
    random.shuffle(images)
    split_index = int(len(images) * (1 - val_split))
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Copy images and create label files for training set
    for idx, img in enumerate(tqdm(train_images, desc="Processing training images"), start=1):
        if os.path.exists(os.path.join(images_folder, img)):
            new_img_name = f"{idx}.jpg"
            shutil.copy(os.path.join(images_folder, img), os.path.join(output_folder, 'images', 'train', new_img_name))
            label_file = f"{idx}.txt"
            with open(os.path.join(output_folder, 'labels', 'train', label_file), 'w') as f:
                for ann in annotations[img]:
                    f.write(" ".join(map(str, ann)) + "\n")
        else:
            print(f"Image {img} does not exist. Skipping...")

    # Copy images and create label files for validation set
    for idx, img in enumerate(tqdm(val_images, desc="Processing validation images"), start=1):
        if os.path.exists(os.path.join(images_folder, img)):
            new_img_name = f"{len(train_images) + idx}.jpg"
            shutil.copy(os.path.join(images_folder, img), os.path.join(output_folder, 'images', 'val', new_img_name))
            label_file = f"{len(train_images) + idx}.txt"
            with open(os.path.join(output_folder, 'labels', 'val', label_file), 'w') as f:
                for ann in annotations[img]:
                    f.write(" ".join(map(str, ann)) + "\n")
        else:
            print(f"Image {img} does not exist. Skipping...")

# Example usage
convert_coco_to_yolo_segmentation('/content/merged_annotations.json', '/content/dataset/new_cleaned_data', '/content/yolo_data')
