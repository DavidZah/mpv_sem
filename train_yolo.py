from ultralytics import YOLO
from dataloader import DroneDataModule
import torch
from torchvision import transforms

import segmentation_models_pytorch as smp








if __name__ == "__main__":

    print(torch.__version__)
    train_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    root_dir = ('/Users/zah/.cache/kagglehub/datasets/kuantinglai/ntut-4k-drone-photo-dataset-for-human-detection/versions/1/')

    dataloader = DroneDataModule(root_dir, train_transform, val_transform, batch_size=1, num_workers=0)
    dataloader.setup()

    model = YOLO("yolo11n.pt")
    model.train(dataloader, epochs=10)

