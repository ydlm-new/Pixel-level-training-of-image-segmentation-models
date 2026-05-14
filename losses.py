import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes=8, smooth=1e-5, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        pred: (B, C, H, W) raw logits
        target: (B, H, W) integer class labels
        """
        pred = F.softmax(pred, dim=1)

        # Create mask for valid pixels
        valid_mask = (target != self.ignore_index)
        target_masked = target.clone()
        target_masked[~valid_mask] = 0

        # One-hot encode target: (B, H, W) -> (B, C, H, W)
        target_onehot = F.one_hot(target_masked, self.num_classes)  # (B, H, W, C)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
        pred = pred * valid_mask
        target_onehot = target_onehot * valid_mask

        # Compute Dice per class
        dims = (0, 2, 3)
        intersection = (pred * target_onehot).sum(dim=dims)
        union = pred.sum(dim=dims) + target_onehot.sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        return dice_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=8, ignore_index=255, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice
