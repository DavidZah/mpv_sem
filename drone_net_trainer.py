import argparse
import torch
import lightning.pytorch as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from dataloader_segmentation import SegmentationDataModule
from models.drone_net import PatchTransformerDETR
from lightning.pytorch import LightningModule
from torch import nn
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

import torch
from torch import nn
import lightning.pytorch as L


import torch
import torch.nn as nn
import torchvision.ops as ops

import torch
import torch.nn as nn
import torchvision.ops as ops

class DroneBBoxPredictionModel(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3, lambda_coord=1.0, lambda_class=1.0, iou_threshold=0.5):
        super(DroneBBoxPredictionModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.lambda_coord = lambda_coord
        self.lambda_class = lambda_class
        self.iou_threshold = iou_threshold

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, pred_bboxes, gt_bboxes, pred_scores=None, gt_labels=None):
        """
        Compute composite loss for bounding box prediction and classification.
        """
        print(f"pred_bboxes shape: {pred_bboxes.shape}")
        print(f"gt_bboxes shape: {gt_bboxes.shape}")
        if pred_scores is not None:
            print(f"pred_scores shape: {pred_scores.shape}")
        if gt_labels is not None:
            print(f"gt_labels shape: {gt_labels.shape}")

        # Match predicted boxes to ground truth using IoU
        iou_matrix = ops.box_iou(pred_bboxes, gt_bboxes)
        matched_indices = torch.argmax(iou_matrix, dim=1)  # Find the best match for each prediction
        matched_gt_bboxes = gt_bboxes[matched_indices]

        # IoU Loss
        iou_loss = 1 - iou_matrix.max(dim=1).values.mean()  # Max IoU per predicted box

        # Regression Loss (Smooth L1)
        regression_loss = nn.SmoothL1Loss()(pred_bboxes, matched_gt_bboxes)

        # Classification Loss (Optional)
        class_loss = 0.0
        if pred_scores is not None and gt_labels is not None:
            matched_gt_labels = gt_labels[matched_indices]
            class_loss = nn.CrossEntropyLoss()(pred_scores, matched_gt_labels)

        # Combined Loss
        total_loss = self.lambda_coord * regression_loss + iou_loss + self.lambda_class * class_loss
        return total_loss

    def apply_nms(self, pred_bboxes, pred_scores, pred_classes):
        """
        Apply Non-Maximum Suppression (NMS) to reduce redundant bounding boxes.
        """
        # Flatten batch dimension
        pred_bboxes = pred_bboxes.view(-1, 4)  # Shape: [batch_size * 128, 4]
        pred_scores = pred_scores.view(-1)  # Shape: [batch_size * 128]
        pred_classes = pred_classes.view(-1)  # Shape: [batch_size * 128]

        # Apply batched NMS
        nms_indices = ops.batched_nms(pred_bboxes, pred_scores, pred_classes, self.iou_threshold)

        # Select filtered boxes, scores, and classes
        filtered_boxes = pred_bboxes[nms_indices]
        filtered_scores = pred_scores[nms_indices]
        filtered_classes = pred_classes[nms_indices]

        return filtered_boxes, filtered_scores, filtered_classes

    def training_step(self, batch, batch_idx):
        images,gt_bboxes = batch
        gt_bboxes = gt_bboxes[0]
        gt_labels = torch.ones(gt_bboxes.shape[0], dtype=torch.float).to(gt_bboxes.device)
        pred_bboxes, pred_scores = self(images)

        loss = self.compute_loss(pred_bboxes, gt_bboxes, pred_scores, gt_labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, gt_bboxes = batch
        gt_bboxes = gt_bboxes[0]
        gt_labels = torch.ones(gt_bboxes.shape[0], dtype=torch.float).to(gt_bboxes.device)

        pred_bboxes, pred_scores = self(images)
        pred_classes = torch.argmax(pred_scores, dim=-1)

        # Apply NMS
        filtered_bboxes, filtered_scores, filtered_classes = self.apply_nms(pred_bboxes, pred_scores, pred_classes)

        if filtered_bboxes.shape[0] == 0:  # No predictions
            print("No predictions after NMS.")
            return {"val_loss": 0.0, "val_mean_iou": 0.0}

        # Compute loss
        loss = self.compute_loss(filtered_bboxes, gt_bboxes, filtered_scores, gt_labels)

        # IoU Metric
        if gt_bboxes.shape[0] > 0:  # Only compute IoU if ground truth exists
            iou_scores = ops.box_iou(filtered_bboxes, gt_bboxes).diagonal()
            mean_iou = iou_scores.mean().item()
        else:
            mean_iou = 0.0

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mean_iou', mean_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_mean_iou": mean_iou}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer



if __name__ == "__main__":
    # Model and Training Configuration
    wandb_project = "DroneDetection"
    experiment_name = "default_experiment"
    log_dir = "logs/"
    data_root = "data/1/ntut_drone_train"
    split_path = "split_indices.json"
    checkpoint_dir = "checkpoints/"
    batch_size = 1
    num_epochs = 20
    learning_rate = 1e-3
    num_workers = 4
    ram_cache = False
    feature_dim = 256
    num_queries = 128
    patch_size = (270, 480)
    image_size = (3840, 2160)

    # WandB Logger
    wandb_logger = WandbLogger(project=wandb_project, name=experiment_name, log_model="all")

    # Data Transformations
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Data Module Initialization
    datamodule = SegmentationDataModule(
        root_dir=data_root,
        split_path=split_path,
        patch_size=None,
        overlap=0.1,
        transform={"train": train_transforms, "val": val_transforms},
        batch_size=batch_size,
        num_workers=num_workers,
        ram_cache=ram_cache,
    )
    datamodule.setup()

    # Initialize the Model
    model = PatchTransformerDETR(
        patch_size=patch_size,
        feature_dim=feature_dim,
        num_queries=num_queries,
        num_classes=1,
        image_size=image_size,
    )

    # Wrap the Model with Lightning
    drone_model = DroneBBoxPredictionModel(model=model, learning_rate=learning_rate)

    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename="best-drone-model",
        save_top_k=1,
        mode="min",
        verbose=True,
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=num_epochs,
        logger=wandb_logger,
        log_every_n_steps=10,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        callbacks=[early_stopping, checkpoint_callback],
    )

    # Train the Model
    trainer.fit(drone_model, datamodule.train_dataloader(), datamodule.val_dataloader())

    # Save the Best Model Path
    best_model_path = checkpoint_callback.best_model_path
    print(f"Training complete. Best model saved at {best_model_path}")
