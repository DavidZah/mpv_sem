import argparse
import torch
import yaml
import timm
import lightning.pytorch as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from timm.data import resolve_model_data_config, create_transform
from dataloader import DroneDataClasificationModule
from torch import nn
import wandb


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
        labels = labels.unsqueeze(1).float()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a Drone Classification Model")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load YAML configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    wandb_logger = WandbLogger(project=config['wandb_project'], log_model='all')
    root_dir = config['data_root']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    model_name = config['model']

    # Initialize the timm model
    model = timm.create_model(model_name, pretrained=True, num_classes=1)

    # Automatic transforms based on timm model requirements
    data_config = resolve_model_data_config(model.pretrained_cfg)

    train_transforms = create_transform(**data_config, is_training=True)
    val_transforms = create_transform(**data_config, is_training=False)

    # Initialize DataLoader with the updated transforms
    dataloader = DroneDataClasificationModule(
        root_dir,
        train_transforms,
        val_transforms,
        batch_size=batch_size,
        overlap_train=0,
        overlap_val=0.1,
        num_workers=16,
        prefetch_factor=3,
        pin_memory=True,
        ram_cache=True
    )
    dataloader.setup()
    train_dataloader, val_dataloader = dataloader.train_dataloader(), dataloader.val_dataloader()

    # Ensure no prior WandB run is active
    if wandb.run is not None:
        wandb.finish()

    wandb.init(project=config['wandb_project'], name=f"{model_name}", reinit=True)

    try:
        # Initialize LightningModule
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
        print(f"Model {model_name} training complete. Best model saved at: {best_model_path}")

    except Exception as e:
        print(f"Training failed for model {model_name}: {e}")

    finally:
        wandb.finish()
