import os
import json
import csv
from collections import defaultdict

import cv2
from matplotlib import pyplot as plt, patches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from random import shuffle


class SegmentationDroneDataset(Dataset):
    def __init__(self, root_dir, patch_size=512, overlap=0.1, transform=None, image_paths=None, ram_cache=True):
        """
        Dataset for segmentation tasks with patches and bounding boxes.

        Args:
            root_dir (string): Root directory containing images and annotations.
            patch_size (int): Size of each patch (default: 512).
            overlap (float): Overlap fraction between patches (default: 0.1).
            transform (callable, optional): Optional transform to apply on patches.
            image_paths (list, optional): List of image paths to include in this dataset. If None, use all images.
            ram_cache (bool): If True, preload all images into memory.
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = transform
        self.ram_cache = ram_cache
        self.data = []  # Stores image metadata
        self.patches = []  # Stores generated patches
        self.images = {} if ram_cache else None

        self._load_data(image_paths)

    def _load_data(self, image_paths):
        """
        Load image paths and corresponding bounding box annotations.
        """
        csv_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_paths.append(os.path.join(root, file))

        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found in {self.root_dir}. Check the directory structure.")

        for csv_path in csv_paths:
            img_dir = os.path.dirname(csv_path)
            with open(csv_path) as csvfile:
                reader = csv.DictReader(csvfile)
                grouped_annotations = defaultdict(list)

                for row in reader:

                    img_path = os.path.join(img_dir, row['image'])


                    if os.path.exists(img_path):
                        bbox = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
                        grouped_annotations[img_path].append(bbox)
                    elif os.path.exists(os.path.join(img_path,"jpg")):
                        bbox = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
                        grouped_annotations[img_path].append(bbox)
                    else:
                        print(f"Image not found: {img_path}")

                for img_path, bboxes in grouped_annotations.items():
                    if image_paths and img_path not in image_paths:
                        continue
                    self.data.append({"img_path": img_path, "bboxes": bboxes})
                    if self.ram_cache:
                        try:
                            image = Image.open(img_path).convert("RGB")
                            self.images[img_path] = np.array(image)
                            image.close()
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")

        if not self.data:
            raise ValueError("No valid images loaded. Check CSV files and image paths.")

        self._generate_patches()

    def _generate_patches(self):
        """
        Generate patches for all images in the dataset.
        """
        stride = int(self.patch_size * (1 - self.overlap))

        for entry in self.data:
            img_path = entry["img_path"]
            bboxes = entry["bboxes"]

            if self.ram_cache:
                image = self.images[img_path]
            else:
                image = np.array(Image.open(img_path).convert("RGB"))

            img_height, img_width = image.shape[:2]

            for y in range(0, img_height, stride):
                for x in range(0, img_width, stride):
                    x_max = min(x + self.patch_size, img_width)
                    y_max = min(y + self.patch_size, img_height)
                    x_min, y_min = x, y

                    patch_bboxes = []
                    for bbox in bboxes:
                        if self._is_bbox_in_patch([x_min, y_min, x_max, y_max], bbox):
                            patch_bboxes.append(self._normalize_bbox([x_min, y_min, x_max, y_max], bbox))

                    self.patches.append((img_path, [x_min, y_min, x_max, y_max], patch_bboxes))

    def _is_bbox_in_patch(self, patch_coords, bbox):
        """
        Check if a bounding box overlaps with a patch such that
        more than 10% of the bounding box is within the patch.
        """
        px_min, py_min, px_max, py_max = patch_coords
        bx_min, by_min, bx_max, by_max = bbox

        # Calculate the intersection coordinates
        inter_min_x = max(px_min, bx_min)
        inter_min_y = max(py_min, by_min)
        inter_max_x = min(px_max, bx_max)
        inter_max_y = min(py_max, by_max)

        # Check if there is an overlap
        if inter_min_x >= inter_max_x or inter_min_y >= inter_max_y:
            return False  # No intersection

        # Calculate intersection area
        intersection_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)

        # Calculate the bounding box area
        bbox_area = (bx_max - bx_min) * (by_max - by_min)

        # Check if more than 10% of the bbox is within the patch
        return intersection_area > 0.1 * bbox_area

    def _normalize_bbox(self, patch_coords, bbox):
        """
        Normalize bounding box coordinates to the patch context.
        """
        px_min, py_min, _, _ = patch_coords
        bx_min, by_min, bx_max, by_max = bbox
        return [
            bx_min - px_min,
            by_min - py_min,
            bx_max - px_min,
            by_max - py_min,
        ]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_path, patch_coords, bboxes = self.patches[idx]
        x_min, y_min, x_max, y_max = patch_coords

        if self.ram_cache:
            image = self.images[img_path]
        else:
            image = np.array(Image.open(img_path).convert("RGB"))

        patch = image[y_min:y_max, x_min:x_max]
        patch = Image.fromarray(patch)

        if self.transform:
            patch = self.transform(patch)

        bboxes = torch.tensor(bboxes, dtype=torch.float32)

        return patch, bboxes


class SegmentationDataModule:
    def __init__(self, root_dir, split_path, patch_size=512, overlap=0.1, transform=None, batch_size=4, num_workers=4,ram_cache=False):
        """
        DataModule for segmentation tasks.

        Args:
            root_dir (string): Root directory containing images and annotations.
            split_path (string): Path to the JSON file containing train/val splits.
            patch_size (int): Size of each patch (default: 512).
            overlap (float): Overlap fraction between patches (default: 0.1).
            transform (callable): Transformations to apply on patches.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
        """
        self.root_dir = root_dir
        self.split_path = split_path
        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ram_cache = ram_cache

    def setup(self):
        if os.path.exists(self.split_path):
            with open(self.split_path, "r") as f:
                splits = json.load(f)
                train_paths = splits["train"]
                val_paths = splits["val"]
        else:
            # Create new splits
            all_data = []
            for root, _, files in os.walk(self.root_dir):
                for file in files:
                    if file.endswith(".jpg") or file.endswith(".png"):
                        all_data.append(os.path.join(root, file))
            shuffle(all_data)
            train_size = int(0.8 * len(all_data))
            train_paths = all_data[:train_size]
            val_paths = all_data[train_size:]
            with open(self.split_path, "w") as f:
                json.dump({"train": train_paths, "val": val_paths}, f)

        self.train_dataset = SegmentationDroneDataset(
            root_dir=self.root_dir,
            patch_size=self.patch_size,
            overlap=self.overlap,
            transform=self.transform,
            image_paths=train_paths,
            ram_cache=self.ram_cache
        )
        self.val_dataset = SegmentationDroneDataset(
            root_dir=self.root_dir,
            patch_size=self.patch_size,
            overlap=self.overlap,
            transform=self.transform,
            image_paths=val_paths,
            ram_cache=self.ram_cache
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    root_dir = 'data/1/ntut_drone_train'
    split_path = 'split_indices.json'

    datamodule = SegmentationDataModule(
        root_dir=root_dir,
        split_path=split_path,
        patch_size=512,
        overlap=0.1,
        transform=transform,
        batch_size=1,
        num_workers=0,
        ram_cache=False
    )
    datamodule.setup()

    for batch in datamodule.train_dataloader():
        imgs, bboxes = batch
        # Convert the image tensor to numpy format
        img = imgs[0].permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)

        # Create a matplotlib figure
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Plot the bounding boxes
        for bbox in bboxes[0]:
            x_min, y_min, x_max, y_max = bbox
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            width = x_max - x_min
            height = y_max - y_min
            # Create a rectangle patch
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)

        # Show the plot
        plt.show()


