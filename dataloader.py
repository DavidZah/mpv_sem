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
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


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

from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import json
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import json
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm



class ClassificationDroneDataset(Dataset):
    def __init__(self, root_dir, patch_size=512, overlap=0.1, bbox_threshold=10, transform=None, ram_cache=True, split=None):
        """
        Dataset for splitting large images into patches for classification.

        Args:
            root_dir (string): Root directory containing images and CSV annotations.
            patch_size (int): Size of each patch (default: 512).
            overlap (float): Overlap fraction between patches (default: 0.1).
            bbox_threshold (float): Minimum percentage of a bounding box within the patch to classify as positive (default: 10%).
            transform (callable, optional): Optional transform to be applied on a patch.
            ram_cache (bool): If True, preload all images into memory. If False, load images on demand.
            split (list, optional): List of image paths to include in this dataset. If None, use all images.
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.overlap = overlap
        self.bbox_threshold = bbox_threshold / 100  # Convert percentage to fraction
        self.transform = transform
        self.ram_cache = ram_cache
        self.images = {} if ram_cache else None
        self.data = defaultdict(list)  # Group bounding boxes by image path
        self.patches = []
        self.split = set(split) if split else None
        self._scan_folders()

    def _scan_folders(self):
        # Step 1: Collect all CSV files
        csv_paths = [os.path.join(root, file) for root, _, files in os.walk(self.root_dir) for file in files if file.endswith(".csv")]

        # Step 2: Extract image paths and bounding boxes
        for csv_path in tqdm(csv_paths, desc="Processing CSV files"):
            self._process_csv(csv_path)

        # Step 3: Filter by split, if provided
        if self.split:
            self.data = {img_path: bboxes for img_path, bboxes in self.data.items() if img_path in self.split}

        # Step 4: Preload images into RAM (optional)
        if self.ram_cache:
            self._preload_images()

        # Step 5: Generate patches for all images
        self._generate_all_patches()

    def _process_csv(self, csv_path):
        """Extract image paths and bounding boxes from a single CSV file."""
        img_dir = os.path.dirname(csv_path)
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                img_path = os.path.join(img_dir, row["image"])
                if not os.path.exists(img_path):
                    img_path += ".jpg"
                if os.path.exists(img_path):
                    bbox = [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]
                    self.data[img_path].append(bbox)

    def _preload_images(self):
        """Preload images into RAM for faster access."""
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._load_image, img_path): img_path for img_path in self.data.keys()}
            for future in tqdm(futures, desc="Preloading Images"):
                img_path, img_array = future.result()
                if img_array is not None:
                    self.images[img_path] = img_array

    def _load_image(self, img_path):
        """Load a single image as a NumPy array."""
        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            img.close()
            return img_path, img_array
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return img_path, None

    def _generate_all_patches(self):
        """Generate patches for all images."""
        for img_path, bboxes in tqdm(self.data.items(), desc="Generating Patches"):
            if self.ram_cache:
                img_array = self.images[img_path]
                img_height, img_width = img_array.shape[:2]
            else:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size

            patches = self._generate_patches(img_path, img_height, img_width, bboxes)
            self.patches.extend(patches)

    def _generate_patches(self, img_path, img_height, img_width, bboxes):
        """Generate patches for a single image."""
        stride = int(self.patch_size * (1 - self.overlap))
        x_coords = np.arange(0, img_width, stride)
        y_coords = np.arange(0, img_height, stride)

        return [
            (img_path, (x, y, min(x + self.patch_size, img_width), min(y + self.patch_size, img_height)), bboxes)
            for y in y_coords for x in x_coords
        ]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_path, patch_coords, bboxes = self.patches[idx]
        if self.ram_cache:
            image = self.images[img_path]
        else:
            image = np.array(Image.open(img_path).convert("RGB"))

        # Extract the patch
        x_min, y_min, x_max, y_max = patch_coords
        patch = image[y_min:y_max, x_min:x_max]
        patch = Image.fromarray(patch)

        # Determine patch label
        patch_label = 0
        for bbox in bboxes:
            if self._is_bbox_in_patch(patch_coords, bbox):
                patch_label = 1
                break

        if self.transform:
            patch = self.transform(patch)

        return patch, patch_label

    def _is_bbox_in_patch(self, patch_coords, bbox):
        px_min, py_min, px_max, py_max = patch_coords
        bx_min, by_min, bx_max, by_max = bbox

        # Calculate intersection
        inter_x_min = max(px_min, bx_min)
        inter_y_min = max(py_min, by_min)
        inter_x_max = min(px_max, bx_max)
        inter_y_max = min(py_max, by_max)

        if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
            return False

        # Calculate areas
        intersection_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        bbox_area = (bx_max - bx_min) * (by_max - by_min)

        return intersection_area / bbox_area >= self.bbox_threshold

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
        pass

    for batch in val_loader:
        pass