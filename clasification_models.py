import argparse
import time
from functools import partial
from random import sample, random, choice
import lightning.pytorch as L
import pandas as pd
import torch
import yaml
from fontTools.merge import timer
import gc  # Add garbage collection to cleanup resources

import wandb
from torch import nn
from torchvision.models import get_model
from torchvision.transforms import transforms
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from dataloader import DroneDataClasificationModule
from torchvision.models import list_models
from models.ghostnet_v2 import ghostnetv2, ghostnetv2_1x
# Iterate over each model
import gc  # Add garbage collection to cleanup resources


class DroneClassificationModel(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super(DroneClassificationModel, self).__init__()
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()  # Suitable for binary classification
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.unsqueeze(1).float()  # Ensure labels are the correct shape for BCEWithLogitsLoss
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.unsqueeze(1).float()  # Ensure labels are the correct shape for BCEWithLogitsLoss
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Log specific samples
        if batch_idx == 0:  # Log only for the first batch in each validation epoch
            true_samples = []
            false_samples = []

            # Collect all possible true and false samples
            for idx in range(len(images)):
                label = labels[idx].item()
                pred = torch.sigmoid(outputs[idx]).item()  # Convert logits to probabilities
                pred_label = 1 if pred > 0.5 else 0  # Binary classification threshold

                if label == 1:  # True label is positive
                    true_samples.append((images[idx], label, pred_label, pred))
                elif label == 0:  # True label is negative
                    false_samples.append((images[idx], label, pred_label, pred))

            # Randomly select one true and one false sample (if available)
            true_sample = choice(true_samples) if true_samples else None
            false_sample = choice(false_samples) if false_samples else None

            # Log the selected samples
            for sample_data, title in zip([true_sample, false_sample], ["True Label", "False Label"]):
                if sample_data is not None:
                    image, label, pred_label, pred = sample_data
                    # Denormalize image for visualization
                    image = image.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
                    mean = [0.485, 0.456, 0.406]
                    std = [0.229, 0.224, 0.225]
                    image = image * std + mean  # Denormalize
                    image = (image * 255).clip(0, 255).astype("uint8")  # Scale to [0, 255]

                    caption = f"Ground Truth: {int(label)}, Prediction: {pred_label} ({pred:.2f})"
                    wandb.log({f"{title} Validation Sample": [wandb.Image(image, caption=caption)]})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a Drone Classification Model")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    wandb_logger = WandbLogger(project=config['wandb_project'], log_model='all')
    root_dir = config['data_root']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    model_name = config['model']

    # Define root directory and transformations
    root_dir = 'data/1/ntut_drone_train'
    train_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    torch.set_float32_matmul_precision('medium')  # Set matmul precision to medium


    dataloader = DroneDataClasificationModule(
        root_dir, train_transforms, val_transforms, batch_size=batch_size, overlap_train=0, overlap_val=0.1, num_workers=16,
        prefetch_factor=3,
        pin_memory=True, ram_cache=True
    )
    dataloader.setup()
    train_dataloader, val_dataloader = dataloader.train_dataloader(), dataloader.val_dataloader()
    print(f"Starting training for model: {model_name}")

    # Ensure no prior WandB run is active
    if wandb.run is not None:
        wandb.finish()

    wandb.init(project=config['wandb_project'], name=f"{model_name}", reinit=True)

    try:
        # Model initialization
        if model_name == "ghostnetv2":
            model = ghostnetv2_1x(num_classes=1)
        else:
            model = get_model(model_name, weights=None, num_classes=1)

        drone_model = DroneClassificationModel(model)

        # Callbacks and Trainer
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=True)
        model_checkpoint = ModelCheckpoint(
            monitor='val_loss', dirpath=f'checkpoints/{model_name}/', filename='best-drone-classification',
            save_top_k=3, mode='min', verbose=True
        )

        trainer = L.Trainer(
            max_epochs=num_epochs,
            logger=wandb_logger,
            log_every_n_steps=10,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            callbacks=[early_stopping, model_checkpoint]
        )

        trainer.fit(drone_model, train_dataloader, val_dataloader)

        best_model_path = model_checkpoint.best_model_path
        val_loss = trainer.callback_metrics['val_loss'].item()
        model_results.append({'Model': model_name, 'Val_Loss': val_loss, 'Best_Model_Path': best_model_path})

        print(f"Model {model_name} training complete. Best model saved at: {best_model_path}")

    except Exception as e:
        print(f"Training failed for model {model_name}: {e}")

    finally:
        wandb.finish()

