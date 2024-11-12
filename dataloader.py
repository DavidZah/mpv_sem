from concurrent.futures import ThreadPoolExecutor

import torchvision.transforms.functional
import csv
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import lightning as L
from random import shuffle, random

import json
import os
import csv
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import numpy as np
from PIL import Image
from torch.utils.data import Dataset



def save_split_indices(split_path, train_indices, val_indices):
    """Save train and validation indices to a JSON file."""
    split_data = {
        "train": train_indices.indices,
        "val": val_indices.indices
    }
    with open(split_path, "w") as f:
        json.dump(split_data, f)


def load_split_indices(split_path):
    """Load train and validation indices from a JSON file."""
    with open(split_path, "r") as f:
        return json.load(f)


class RecursiveDroneDataset(Dataset):
    def __init__(self, root_dir, transform=None, format="default"):
        """
        Args:
            root_dir (string): Root directory containing nested subfolders with images and CSV annotations.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.label_map = {}
        self.format = format
        # Recursively scan all folders to find CSV files
        self._scan_folders(root_dir)

    def _scan_folders(self, folder):
        """
        Recursively scan folders to find CSV files and load annotations.

        Args:
            folder (string): Current folder to scan for CSV files and images.
        """
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(root, file)
                    self._load_annotations(csv_path, root)

    def _load_annotations(self, csv_path, img_dir):
        """
        Load annotations from a CSV file and append to the dataset.

        Args:
            csv_path (string): Path to the CSV file.
            img_dir (string): Directory containing images referenced in the CSV file.
        """
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            annotations = list(reader)

        # Update label map with any new labels in this CSV
        unique_labels = set(row["label"] for row in annotations)
        for label in unique_labels:
            if label not in self.label_map:
                self.label_map[label] = len(self.label_map)

        # Group annotations by image path
        grouped_annotations = defaultdict(list)
        for row in annotations:
            img_path = os.path.join(img_dir, f"{row['image']}")
            if os.path.exists(img_path):
                bbox = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
                label = self.label_map[row["label"]]
                grouped_annotations[img_path].append((bbox, label))

        if self.format == "default":
            # Add each grouped annotation to the dataset
            for img_path, bboxes_labels in grouped_annotations.items():
                bboxes, labels = zip(*bboxes_labels)
                self.data.append({
                    "img_path": img_path,
                    "bboxes": bboxes,
                    "labels": labels
                })

        elif self.format == "coco":
            for img_path, bboxes_labels in grouped_annotations.items():
                # coco format requires  x_center y_center width height format

                bboxes, labels = zip(*bboxes_labels)
                coco_bboxes = []
                for bbox in bboxes:
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    coco_bboxes.append([x_center, y_center, width, height])
                self.data.append({
                    "img_path": img_path,
                    "bboxes": coco_bboxes,
                    "labels": labels
                })
        else:
            raise ValueError("Invalid format argument. Must be 'default' or 'coco format'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        img_path = entry["img_path"]
        bboxes = entry["bboxes"]
        labels = entry["labels"]

        image = Image.open(img_path).convert("RGB")

        # Normalize bbox coordinates to [0, 1]
        width, height = image.size
        normalized_bboxes = []
        for bbox in bboxes:
            normalized_bbox = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
            normalized_bboxes.append(normalized_bbox)

        bboxes = torch.tensor(normalized_bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        target = {"boxes": bboxes, "labels": labels}
        return image, target


from torch.utils.data import DataLoader, random_split
import lightning as L
from torch.utils.data import DataLoader, random_split


class DroneDataModule(L.LightningDataModule):
    def __init__(self, root_dir, train_transform=None, val_transform=None, batch_size=1, num_workers=0):
        super().__init__()
        self.root_dir = root_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Create the full training dataset with the specified train transform
        full_train_dataset = RecursiveDroneDataset(
            root_dir=f"{self.root_dir}/ntut_drone_train",
            transform=self.train_transform
        )

        # Split the full training dataset into training and validation subsets
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )

        # Update the transform for the validation dataset
        self.val_dataset.dataset.transform = self.val_transform

        # Create the test dataset with the validation transform
        self.test_dataset = RecursiveDroneDataset(
            root_dir=f"{self.root_dir}/ntut_drone_test",
            transform=self.val_transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class ClassificationDroneDataset(Dataset):
    def __init__(self, root_dir, patch_size=512, overlap=0.1, transform=None, ram_cache=True, split=None):
        """
        Dataset for splitting large images into patches for classification.

        Args:
            root_dir (string): Root directory containing images and CSV annotations.
            patch_size (int): Size of each patch (default: 512).
            overlap (float): Overlap fraction between patches (default: 0.1).
            transform (callable, optional): Optional transform to be applied on a patch.
            ram_cache (bool): If True, preload all images into memory. If False, load images on demand.
            split (list, optional): List of image paths to include in this dataset. If None, use all images.
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = transform
        self.ram_cache = ram_cache
        self.images = {} if ram_cache else None  # Preloaded images in memory if ram_cache is True
        self.data = []  # Stores (image_path, bboxes)
        self.patches = []  # Stores flattened list of all patches
        self.split = split  # Optional: list of image paths to include
        self._scan_folders()

    def _scan_folders(self):
        """Recursively scans folders for CSV annotations and images."""
        csv_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_paths.append(os.path.join(root, file))

        # Use multiprocessing to speed up annotation loading
        with Pool(cpu_count()) as pool:
            results = pool.map(self._load_annotations, csv_paths)

        # Combine results
        for image_data, patch_data in results:
            if self.ram_cache:
                self.images.update(image_data)
            self.data.extend(patch_data)

        # Filter data by split, if provided
        if self.split is not None:
            self.data = [entry for entry in self.data if entry["img_path"] in self.split]

        # Precompute all patches
        self._generate_all_patches()

    def _load_annotations(self, csv_path):
        """Loads annotations and optionally preloads images for a single CSV."""
        img_dir = os.path.dirname(csv_path)
        image_data = {}
        patch_data = []

        # Load annotations
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            annotations = list(reader)

        grouped_annotations = defaultdict(list)
        for row in annotations:
            img_path = os.path.join(img_dir, row['image'])
            if os.path.exists(img_path):
                bbox = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
                grouped_annotations[img_path].append(bbox)

        for img_path, bboxes in grouped_annotations.items():
            if self.ram_cache:
                # Preload and preprocess images
                if img_path not in image_data:
                    image = Image.open(img_path).convert("RGB")
                    image_data[img_path] = np.array(image)  # Convert to numpy array for faster slicing
                    image.close()

            patch_data.append({
                "img_path": img_path,
                "bboxes": bboxes
            })

        return image_data, patch_data

    def _generate_patches(self, img_path, img_height, img_width, bboxes):
        """Generates patches for a single image."""
        stride = int(self.patch_size * (1 - self.overlap))

        # Precompute x and y ranges using numpy
        x_coords = np.arange(0, img_width, stride)
        y_coords = np.arange(0, img_height, stride)

        # Generate all combinations of patch coordinates
        patches = [
            (img_path, (x, y, min(x + self.patch_size, img_width), min(y + self.patch_size, img_height)), bboxes)
            for y in y_coords for x in x_coords
        ]

        return patches

    def _generate_all_patches(self):
        """Precompute all patches for all images in the dataset."""
        self.patches = []

        image = np.array(Image.open(self.data[0]["img_path"]).convert("RGB"))
        img_height, img_width = image.shape[:2]

        for entry in self.data:
            img_path = entry["img_path"]
            bboxes = entry["bboxes"]
            self.patches.extend(self._generate_patches(img_path, img_height, img_width, bboxes))

    def set_overlap(self, overlap):
        """Set a new overlap value and regenerate patches."""
        self.overlap = overlap
        self._generate_all_patches()

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_path, patch_coords, bboxes = self.patches[idx]

        # Load image: from RAM if cached, else from disk
        if self.ram_cache:
            image = self.images[img_path]
        else:
            image = np.array(Image.open(img_path).convert("RGB"))

        # Extract the patch
        x_min, y_min, x_max, y_max = patch_coords
        patch = image[y_min:y_max, x_min:x_max]

        # Convert patch back to PIL image for transforms
        patch = Image.fromarray(patch)

        # Determine if any bounding box overlaps with this patch
        patch_label = 0  # Default: no object
        for bbox in bboxes:
            if self._is_bbox_in_patch(patch_coords, bbox):
                patch_label = 1
                break

        if self.transform:
            patch = self.transform(patch)

        return patch, patch_label

    def _is_bbox_in_patch(self, patch_coords, bbox):
        """
        Check if a bounding box overlaps with a patch.

        Args:
            patch_coords (tuple): (x_min, y_min, x_max, y_max) of the patch.
            bbox (list): [x_min, y_min, x_max, y_max] of the bounding box.

        Returns:
            bool: True if the bounding box overlaps with the patch, False otherwise.
        """
        patch_xmin, patch_ymin, patch_xmax, patch_ymax = patch_coords
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox
        return not (bbox_xmax <= patch_xmin or bbox_xmin >= patch_xmax or
                    bbox_ymax <= patch_ymin or bbox_ymin >= patch_ymax)

    @staticmethod
    def save_split(split_path, train_list, val_list):
        """Save train and validation splits as JSON files."""
        with open(split_path, "w") as f:
            json.dump({"train": train_list, "val": val_list}, f)

    @staticmethod
    def load_split(split_path):
        """Load train and validation splits from a JSON file."""
        with open(split_path, "r") as f:
            return json.load(f)


class DroneDataClasificationModule(L.LightningDataModule):
    def __init__(self, root_dir, train_transform=None, val_transform=None, batch_size=1, num_workers=0,
                 patch_size=512, overlap_train=0.1, overlap_val=0.1, prefetch_factor=4, pin_memory=True,
                 split_path="split_indices.json", ram_cache=True, shuffle=True,test=False):
        super().__init__()
        self.root_dir = root_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.overlap_train = overlap_train
        self.overlap_val = overlap_val
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.split_path = split_path
        self.ram_cache = ram_cache
        self.shuffle = shuffle
        self.test = test

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # Check if split exists; if not, create and save it
        if os.path.exists(self.split_path):

            splits = ClassificationDroneDataset.load_split(self.split_path)
            if self.test:
                train_images = splits["train"][0:10]
                val_images = splits["val"][0:10]
            else:
                train_images = splits["train"]
                val_images = splits["val"]
        else:
            # Create a temporary dataset to fetch all image paths
            full_dataset = ClassificationDroneDataset(
                root_dir=f"{self.root_dir}/ntut_drone_train",
                patch_size=self.patch_size,
                overlap=0,  # Overlap doesn't matter here
                transform=None,
                ram_cache=False
            )
            all_images = [entry["img_path"] for entry in full_dataset.data]
            shuffle(all_images)
            train_images = all_images[:int(0.8 * len(all_images))]
            val_images = all_images[int(0.8 * len(all_images)):]

            # Save splits
            ClassificationDroneDataset.save_split(self.split_path, train_images, val_images)

        # Create train and validation datasets
        self.train_dataset = ClassificationDroneDataset(
            root_dir=f"{self.root_dir}/ntut_drone_train",
            patch_size=self.patch_size,
            overlap=self.overlap_train,
            transform=self.train_transform,
            ram_cache=self.ram_cache,
            split=train_images,
        )

        self.val_dataset = ClassificationDroneDataset(
            root_dir=f"{self.root_dir}/ntut_drone_train",
            patch_size=self.patch_size,
            overlap=self.overlap_val,
            transform=self.val_transform,
            ram_cache=self.ram_cache,
            split=val_images
        )

    def train_dataloader(self):
        if self.num_workers == 0:
            persistent_workers = False
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=False
        )

    def val_dataloader(self):
        if self.num_workers == 0:
            persistent_workers = False
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=False
        )


if __name__ == '__main__':
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Path to the root directory containing nested subfolders with images and annotations
    root_dir = 'data/1/ntut_drone_train'

    datamodule = DroneDataClasificationModule(root_dir, train_transform=transform, val_transform=transform,
                                              batch_size=1, num_workers=0,patch_size=512, ram_cache=False,prefetch_factor=None)
    datamodule.setup()
    #test val and train dataloader
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    for batch in train_loader:
        print(batch)
        break
    for batch in val_loader:
        print(batch)
        break