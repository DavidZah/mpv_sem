import os
import torch
from torchvision.models import get_model
from torchvision.transforms import transforms
from PIL import Image, ImageDraw
from math import ceil
from tqdm import tqdm
from PIL import ImageFont
from clasification_models import DroneClassificationModel
from models.ghostnet_v2 import ghostnetv2_1x


def load_model_from_checkpoint(checkpoint_path, model_name, device='cpu'):
    """
    Load a PyTorch Lightning model from a checkpoint.

    Args:
        checkpoint_path: Path to the Lightning checkpoint.
        model_name: Name of the model architecture.
        device: Device to load the model onto ('cpu' or 'cuda').

    Returns:
        model: The model loaded with weights.
    """
    # Dynamically instantiate the base model

    if model_name == "ghostnetv2":
        base_model = ghostnetv2_1x(num_classes=1)
    else:
        base_model = get_model(model_name, weights=None, num_classes=1)

    # Load the checkpoint into the LightningModule class
    model = DroneClassificationModel.load_from_checkpoint(
        checkpoint_path,
        model=base_model  # Pass required initialization arguments
    )
    model.to(device)
    model.eval()
    return model





def process_image(model, img_path, patch_size, overlap, threshold, transform):
    """
    Process a single image, classify its patches, and return an annotated image.

    Args:
        model: Trained PyTorch model.
        img_path: Path to the input image.
        patch_size: Size of the patch to extract.
        overlap: Overlap fraction between patches.
        threshold: Classification threshold.
        transform: Transformations to apply to patches.

    Returns:
        Annotated image with detections.
    """
    device = next(model.parameters()).device

    # Load the original image
    original_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(original_image, "RGBA")
    width, height = original_image.size

    stride = int(patch_size * (1 - overlap))
    num_patches_x = ceil((width - patch_size) / stride) + 1
    num_patches_y = ceil((height - patch_size) / stride) + 1

    patches = []
    coords = []

    for i in range(num_patches_y):
        for j in range(num_patches_x):
            x_min = j * stride
            y_min = i * stride
            x_max = min(x_min + patch_size, width)
            y_max = min(y_min + patch_size, height)

            patch = original_image.crop((x_min, y_min, x_max, y_max))
            coords.append((x_min, y_min, x_max, y_max))

            if transform:
                patches.append(transform(patch))

    if not patches:
        return original_image

    # Stack patches into a batch
    patch_tensor_batch = torch.stack(patches).to(device)

    # Make predictions for all patches
    with torch.no_grad():
        outputs = model(patch_tensor_batch)
        probs = torch.sigmoid(outputs).squeeze().cpu().tolist()

    # Draw rectangles for detections exceeding threshold
    for coord, prob in zip(coords, probs):
        if prob > threshold:
            x_min, y_min, x_max, y_max = coord
            draw.rectangle(
                [(x_min, y_min), (x_max, y_max)],
                outline=(0, 255, 0, 255),  # Solid green outline
                fill=(0, 255, 0, 100)  # Semi-transparent green background
            )
            # Write probability text on patch
            draw.text((x_min, y_min), f"{prob:.2f}", fill=(255, 255, 255, 255))

    return original_image


def visualize_detections(model, image_dir, output_dir, patch_size, overlap, threshold, transform):
    """
    Process and annotate all images in a directory.

    Args:
        model: The trained PyTorch model.
        image_dir: Directory containing input images.
        output_dir: Directory to save annotated images.
        patch_size: Size of the patch to extract.
        overlap: Overlap fraction between patches.
        threshold: Classification threshold.
        transform: Transformations to apply to patches.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    for img_file in tqdm(os.listdir(image_dir)):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, img_file)
            annotated_image = process_image(
                model, img_path, patch_size, overlap, threshold, transform
            )
            output_path = os.path.join(output_dir, img_file)
            annotated_image.save(output_path)
            print(f"Saved visualized image: {output_path}")


if __name__ == "__main__":
    # Define the model architecture and checkpoint path
    model_name = "ghostnetv2"  # Example: Use the desired model
    checkpoint_path = "/home/david/mpv_sem/checkpoints/ghostnetv2/best-drone-classification.ckpt"
    dataset_root = "data/1/ntut_drone_test/ntut_drone_test/Drone_004/vott-csv-export"
    output_directory = "output/visualizations"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the trained model
    model = load_model_from_checkpoint(checkpoint_path, model_name, device=device)

    # Transform for input patches
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Visualize detections
    visualize_detections(
        model=model,
        image_dir=dataset_root,
        output_dir=output_directory,
        patch_size=512,
        overlap=0.1,
        threshold=0.5,
        transform=transform
    )
