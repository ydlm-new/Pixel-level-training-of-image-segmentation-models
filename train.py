import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import swanlab

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from model import UNet
from dataset import StanfordBackgroundDataset, get_file_lists
from losses import DiceLoss, CombinedLoss


# ============ Configuration ============
NUM_CLASSES = 8
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
IGNORE_INDEX = 255
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iccv09Data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = ['sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain', 'foreground']


def compute_miou(pred, target, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    """Compute mean Intersection over Union."""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    iou_list = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        if union == 0:
            continue
        iou_list.append(intersection / union)

    return np.mean(iou_list) if iou_list else 0.0


def compute_pixel_accuracy(pred, target, ignore_index=IGNORE_INDEX):
    """Compute pixel-level accuracy."""
    valid = target != ignore_index
    correct = (pred[valid] == target[valid]).sum().item()
    total = valid.sum().item()
    return correct / total if total > 0 else 0.0


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        acc = compute_pixel_accuracy(preds, labels)

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

    return total_loss / num_batches, total_acc / num_batches


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0
    num_batches = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        acc = compute_pixel_accuracy(preds, labels)
        miou = compute_miou(preds, labels)

        total_loss += loss.item()
        total_acc += acc
        total_miou += miou
        num_batches += 1

    return total_loss / num_batches, total_acc / num_batches, total_miou / num_batches


def train(loss_type='ce'):
    print(f"\n{'='*60}")
    print(f"Training with loss: {loss_type}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Dataset
    image_dir = os.path.join(DATA_ROOT, 'images')
    label_dir = os.path.join(DATA_ROOT, 'labels')
    train_files, val_files = get_file_lists(image_dir)

    print(f"Train samples: {len(train_files)}, Val samples: {len(val_files)}")

    train_dataset = StanfordBackgroundDataset(image_dir, label_dir, train_files,
                                              img_size=IMG_SIZE, augment=True)
    val_dataset = StanfordBackgroundDataset(image_dir, label_dir, val_files,
                                            img_size=IMG_SIZE, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    # Model (base_channels=16 for CPU training efficiency)
    model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_channels=16).to(DEVICE)

    # Loss function
    if loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    elif loss_type == 'dice':
        criterion = DiceLoss(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    elif loss_type == 'ce_dice':
        criterion = CombinedLoss(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # SwanLab init (local mode, no login required)
    swanlab.init(
        project="unet-segmentation",
        experiment_name=f"unet_{loss_type}",
        mode="local",
        config={
            "model": "UNet",
            "dataset": "Stanford Background Dataset",
            "num_classes": NUM_CLASSES,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "loss_function": loss_type,
            "weight_decay": 1e-4,
            "train_samples": len(train_files),
            "val_samples": len(val_files),
        }
    )

    best_miou = 0.0
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_miou = validate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val mIoU: {val_miou:.4f}")

        # Log to SwanLab
        swanlab.log({
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/mIoU": val_miou,
            "lr": optimizer.param_groups[0]['lr'],
        })

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_model_{loss_type}.pth'))
            print(f"  -> New best mIoU: {best_miou:.4f}, model saved.")

    print(f"\nTraining complete. Best Val mIoU: {best_miou:.4f}")
    swanlab.finish()

    return best_miou


if __name__ == '__main__':
    loss_type = sys.argv[1] if len(sys.argv) > 1 else 'ce'
    if loss_type not in ['ce', 'dice', 'ce_dice']:
        print("Usage: python train.py [ce|dice|ce_dice]")
        sys.exit(1)
    train(loss_type)
