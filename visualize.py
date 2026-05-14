"""Visualize segmentation predictions from the three models."""
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import UNet
from dataset import StanfordBackgroundDataset, get_file_lists

NUM_CLASSES = 8
IMG_SIZE = (128, 128)
CLASS_NAMES = ['sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain', 'foreground']
COLORS = np.array([
    [128, 128, 255],  # sky - light blue
    [0, 128, 0],      # tree - green
    [128, 128, 128],  # road - gray
    [0, 255, 0],      # grass - bright green
    [0, 0, 255],      # water - blue
    [255, 128, 0],    # building - orange
    [128, 64, 0],     # mountain - brown
    [255, 0, 0],      # foreground - red
], dtype=np.uint8)


def colorize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(NUM_CLASSES):
        color_mask[mask == cls] = COLORS[cls]
    return color_mask


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, 'iccv09Data', 'images')
    label_dir = os.path.join(base_dir, 'iccv09Data', 'labels')
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')

    _, val_files = get_file_lists(image_dir)

    dataset = StanfordBackgroundDataset(image_dir, label_dir, val_files, img_size=IMG_SIZE, augment=False)

    # Load models
    models = {}
    for loss_type in ['ce', 'dice', 'ce_dice']:
        model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_channels=16)
        ckpt_path = os.path.join(checkpoint_dir, f'best_model_{loss_type}.pth')
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.eval()
        models[loss_type] = model

    # Select sample images
    sample_indices = [0, 10, 20, 30, 40]
    fig, axes = plt.subplots(len(sample_indices), 5, figsize=(15, 3 * len(sample_indices)))

    col_titles = ['Input Image', 'Ground Truth', 'CE Loss', 'Dice Loss', 'CE + Dice']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight='bold')

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for row, idx in enumerate(sample_indices):
        img_tensor, label = dataset[idx]

        # Denormalize image for display
        img_np = img_tensor.numpy().transpose(1, 2, 0)
        img_np = (img_np * std + mean).clip(0, 1)

        # Ground truth
        gt_mask = label.numpy()
        gt_colored = colorize_mask(gt_mask)

        axes[row, 0].imshow(img_np)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(gt_colored)
        axes[row, 1].axis('off')

        # Predictions
        with torch.no_grad():
            input_batch = img_tensor.unsqueeze(0)
            for col, loss_type in enumerate(['ce', 'dice', 'ce_dice'], start=2):
                pred = models[loss_type](input_batch)
                pred_mask = pred.argmax(dim=1).squeeze(0).numpy()
                pred_colored = colorize_mask(pred_mask)
                axes[row, col].imshow(pred_colored)
                axes[row, col].axis('off')

    # Add legend
    patches = [mpatches.Patch(color=COLORS[i]/255.0, label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc='lower center', ncol=8, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Segmentation Results Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(base_dir, 'segmentation_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
