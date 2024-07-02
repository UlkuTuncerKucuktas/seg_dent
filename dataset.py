import os
import json
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
import segmentation_models_pytorch as smp
import random
import albumentations as albu
import cv2
from PIL import ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ToSingleChannel(albu.ImageOnlyTransform):
    def apply(self, img, **params):
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray_img

def get_training_augmentation():
    train_transform = [
        #albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.Resize(320, 320, always_apply=True),
        #albu.IAAAdditiveGaussianNoise(p=0.2),
        #albu.IAAPerspective(p=0.5),
        #albu.RandomSizedCrop(min_max_height=(120, 220), height=320, width=320, p=0.25),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
            ],
            p=0.8,
        ),
        
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(320, 320, always_apply=True),
        

    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):

    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class SegmentationDataset(Dataset):
    def __init__(self, annotations_file, img_dir, indices, preprocessing=None, augmentation=None):
        self.img_labels = json.load(open(annotations_file))
        self.img_labels['images'] = [img for img in self.img_labels['images']
                                     if os.path.exists(os.path.join(img_dir, img['file_name']))]
        self.img_dir = img_dir
        self.indices = indices
        self.preprocessing = preprocessing
        self.augmentation = augmentation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        adjusted_idx = self.indices[idx]
        img_path = os.path.join(self.img_dir, self.img_labels['images'][adjusted_idx]['file_name'])
        image = Image.open(img_path).convert('L') 

        image_id = self.img_labels['images'][adjusted_idx]['id']
        mask = self.create_mask(image_id, self.img_labels['annotations'], image.size)

      
        image, mask = self.pad_to_square(image, mask)

        image = np.array(image)
        if image.ndim == 2: 
            image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(np.array(mask), axis=-1)

        original_image = image.copy()

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            original_image = self.augmentation(image=original_image)['image']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        image = np.mean(image, axis=0, keepdims=True)
        original_image = np.mean(original_image, axis=0, keepdims=True)

        return original_image, image, mask

    def create_mask(self, image_id, annotations, image_size):
        mask = Image.new('L', image_size, 0)
        for annotation in annotations:
            if (annotation['image_id'] == image_id):
                for segment in annotation['segmentation']:
                    ImageDraw.Draw(mask).polygon(segment, outline=1, fill=1)
        return mask

    def pad_to_square(self, image, mask):
        width, height = image.size
        if width == height:
            return image, mask
        max_side = max(width, height)
        pad_width = (max_side - width) // 2
        pad_height = (max_side - height) // 2
        image = ImageOps.expand(image, (pad_width, pad_height, pad_width, pad_height), fill=0)
        mask = ImageOps.expand(mask, (pad_width, pad_height, pad_width, pad_height), fill=0)
        return image, mask


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def visualize_with_masks_single(image, mask):
    """Plot a single image with its mask."""
    plt.figure(figsize=(6, 6))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)  # alpha controls the transparency of the mask
    plt.show()

def create_train_val_datasets(annotations_file, img_dir, val_split=0.2, preprocessing_fn=None, seed=42):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


    img_labels = json.load(open(annotations_file))
    img_labels['images'] = [img for img in img_labels['images'] if os.path.exists(os.path.join(img_dir, img['file_name']))]
    total_size = len(img_labels['images'])


    val_size = int(val_split * total_size)
    train_size = total_size - val_size

    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]


    train_dataset = SegmentationDataset(annotations_file, img_dir, train_indices, augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    val_dataset = SegmentationDataset(annotations_file, img_dir, val_indices, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

    return train_dataset, val_dataset
