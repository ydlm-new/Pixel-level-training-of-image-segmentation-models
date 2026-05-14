"""Run all three loss configurations sequentially."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train

if __name__ == '__main__':
    results = {}

    for loss_type in ['ce', 'dice', 'ce_dice']:
        best_miou = train(loss_type)
        results[loss_type] = best_miou

    print("\n" + "=" * 60)
    print("FINAL RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Loss Type':<20} {'Best Val mIoU':<15}")
    print("-" * 35)
    for loss_type, miou in results.items():
        name = {'ce': 'Cross-Entropy', 'dice': 'Dice Loss', 'ce_dice': 'CE + Dice'}[loss_type]
        print(f"{name:<20} {miou:.4f}")
    print("=" * 60)
