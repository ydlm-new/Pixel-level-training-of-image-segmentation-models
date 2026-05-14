"""Generate comparison plots for the three loss configurations."""
import os
import json
import matplotlib.pyplot as plt
import numpy as np

SWANLOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'swanlog')


def load_swanlab_metrics(run_dir):
    """Load metrics from swanlab run directory."""
    metrics = {}
    logs_dir = os.path.join(run_dir, 'logs')
    if not os.path.exists(logs_dir):
        return metrics

    for metric_dir in os.listdir(logs_dir):
        metric_path = os.path.join(logs_dir, metric_dir)
        if os.path.isdir(metric_path):
            values = []
            for f in sorted(os.listdir(metric_path)):
                if f.endswith('.log'):
                    with open(os.path.join(metric_path, f), 'r') as fp:
                        for line in fp:
                            line = line.strip()
                            if line:
                                try:
                                    data = json.loads(line)
                                    if 'data' in data:
                                        values.append(data['data'])
                                    elif 'value' in data:
                                        values.append(data['value'])
                                except json.JSONDecodeError:
                                    pass
            metrics[metric_dir] = values
    return metrics


def main():
    # Manually record results from training output
    results = {
        'ce': {
            'train_loss': [1.7987, 1.3730, 1.1396, 1.0163, 0.9623, 0.8931, 0.8558, 0.8439, 0.8046, 0.8013,
                          0.7727, 0.7449, 0.7159, 0.7125, 0.7004, 0.6804, 0.6761, 0.6581, 0.6518, 0.6340,
                          0.6292, 0.6218, 0.6024, 0.5882, 0.5725, 0.5832, 0.5736, 0.5705, 0.5727, 0.5598],
            'val_loss': [1.5043, 1.1686, 1.1134, 0.9429, 0.9085, 0.8520, 0.8209, 0.7971, 0.8157, 0.7449,
                        0.7529, 0.7213, 0.7192, 0.7065, 0.6891, 0.6917, 0.6568, 0.6478, 0.6553, 0.6267,
                        0.6223, 0.6230, 0.6193, 0.6181, 0.6001, 0.5990, 0.5945, 0.5891, 0.5941, 0.5918],
            'train_acc': [0.3344, 0.5899, 0.6611, 0.6747, 0.6855, 0.7013, 0.7143, 0.7175, 0.7315, 0.7305,
                         0.7411, 0.7505, 0.7624, 0.7619, 0.7650, 0.7754, 0.7757, 0.7818, 0.7841, 0.7895,
                         0.7922, 0.7951, 0.8005, 0.8078, 0.8115, 0.8101, 0.8108, 0.8135, 0.8118, 0.8160],
            'val_acc': [0.4889, 0.6632, 0.6358, 0.6869, 0.6945, 0.7108, 0.7139, 0.7346, 0.7222, 0.7455,
                       0.7502, 0.7544, 0.7565, 0.7635, 0.7676, 0.7590, 0.7803, 0.7827, 0.7792, 0.7911,
                       0.7930, 0.7915, 0.7959, 0.7961, 0.8018, 0.8019, 0.8035, 0.8079, 0.8016, 0.8030],
            'val_miou': [0.2733, 0.4089, 0.3867, 0.4421, 0.4470, 0.4625, 0.4585, 0.4881, 0.4731, 0.4929,
                        0.5113, 0.5044, 0.5073, 0.5208, 0.5245, 0.5187, 0.5446, 0.5347, 0.5305, 0.5583,
                        0.5531, 0.5501, 0.5620, 0.5630, 0.5685, 0.5693, 0.5723, 0.5707, 0.5645, 0.5658],
        },
        'dice': {
            'train_loss': [0.7905, 0.6803, 0.5833, 0.5295, 0.5044, 0.4873, 0.4727, 0.4732, 0.4608, 0.4596,
                          0.4474, 0.4360, 0.4416, 0.4260, 0.4221, 0.4231, 0.4129, 0.4156, 0.4049, 0.3999,
                          0.4046, 0.3846, 0.3833, 0.3926, 0.3776, 0.3708, 0.3682, 0.3709, 0.3631, 0.3637],
            'val_loss': [0.6983, 0.6159, 0.5469, 0.4986, 0.4986, 0.4723, 0.4806, 0.4509, 0.4480, 0.4405,
                        0.4579, 0.4495, 0.4346, 0.4180, 0.4241, 0.4269, 0.4274, 0.4058, 0.4040, 0.4092,
                        0.4053, 0.3937, 0.3921, 0.3806, 0.3943, 0.3833, 0.3773, 0.3770, 0.3754, 0.3747],
            'train_acc': [0.4459, 0.6128, 0.6596, 0.6705, 0.6701, 0.6887, 0.6904, 0.6961, 0.7059, 0.7126,
                         0.7166, 0.7274, 0.7222, 0.7427, 0.7470, 0.7409, 0.7551, 0.7562, 0.7626, 0.7651,
                         0.7631, 0.7716, 0.7774, 0.7728, 0.7769, 0.7781, 0.7786, 0.7819, 0.7879, 0.7863],
            'val_acc': [0.5564, 0.6578, 0.6708, 0.6943, 0.6604, 0.7019, 0.6887, 0.7071, 0.7170, 0.7233,
                       0.7031, 0.7017, 0.7361, 0.7491, 0.7476, 0.7474, 0.7416, 0.7600, 0.7550, 0.7557,
                       0.7551, 0.7713, 0.7670, 0.7807, 0.7614, 0.7734, 0.7773, 0.7773, 0.7811, 0.7811],
            'val_miou': [0.3079, 0.4259, 0.4214, 0.4551, 0.4368, 0.4681, 0.4453, 0.4743, 0.4760, 0.4827,
                        0.4618, 0.4716, 0.4889, 0.5094, 0.5001, 0.4949, 0.4928, 0.5099, 0.4957, 0.4866,
                        0.4922, 0.5054, 0.5083, 0.5214, 0.5043, 0.5158, 0.5223, 0.5219, 0.5244, 0.5253],
        },
        'ce_dice': {
            'train_loss': [2.4408, 1.9852, 1.7485, 1.5675, 1.4715, 1.4121, 1.3579, 1.3296, 1.2684, 1.2555,
                          1.2156, 1.2296, 1.1779, 1.1591, 1.1429, 1.1348, 1.1237, 1.0701, 1.0508, 1.0543,
                          1.0409, 1.0090, 0.9924, 0.9909, 0.9647, 0.9747, 0.9674, 0.9523, 0.9454, 0.9700],
            'val_loss': [1.9832, 1.8872, 1.5501, 1.4968, 1.4264, 1.3574, 1.3242, 1.3696, 1.2207, 1.4062,
                        1.2131, 1.2253, 1.1793, 1.1087, 1.1687, 1.1332, 1.1144, 1.0809, 1.0524, 1.0658,
                        1.0641, 1.0382, 1.0229, 1.0158, 1.0284, 1.0123, 1.0117, 1.0135, 1.0028, 1.0041],
            'train_acc': [0.4702, 0.6135, 0.6568, 0.6923, 0.7073, 0.7165, 0.7209, 0.7313, 0.7459, 0.7478,
                         0.7528, 0.7531, 0.7669, 0.7723, 0.7737, 0.7751, 0.7783, 0.7920, 0.7947, 0.7931,
                         0.7981, 0.8059, 0.8092, 0.8097, 0.8140, 0.8148, 0.8157, 0.8176, 0.8202, 0.8170],
            'val_acc': [0.6185, 0.6323, 0.6998, 0.6941, 0.7098, 0.7174, 0.7280, 0.7148, 0.7515, 0.7004,
                       0.7552, 0.7513, 0.7581, 0.7805, 0.7585, 0.7688, 0.7787, 0.7887, 0.7932, 0.7924,
                       0.7896, 0.7977, 0.8016, 0.8006, 0.7995, 0.8047, 0.8019, 0.8014, 0.8047, 0.8049],
            'val_miou': [0.3223, 0.3646, 0.4522, 0.4567, 0.4613, 0.4730, 0.4887, 0.4658, 0.5037, 0.4494,
                        0.5151, 0.4977, 0.5100, 0.5420, 0.5213, 0.5390, 0.5311, 0.5474, 0.5541, 0.5533,
                        0.5520, 0.5649, 0.5676, 0.5678, 0.5709, 0.5798, 0.5690, 0.5693, 0.5729, 0.5786],
        },
    }

    epochs = list(range(1, 31))
    colors = {'ce': '#1f77b4', 'dice': '#ff7f0e', 'ce_dice': '#2ca02c'}
    labels = {'ce': 'Cross-Entropy', 'dice': 'Dice Loss', 'ce_dice': 'CE + Dice'}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('U-Net Segmentation: Loss Function Comparison\nStanford Background Dataset', fontsize=14, fontweight='bold')

    # Plot 1: Training Loss
    ax = axes[0, 0]
    for key in ['ce', 'dice', 'ce_dice']:
        ax.plot(epochs, results[key]['train_loss'], color=colors[key], label=labels[key], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for key in ['ce', 'dice', 'ce_dice']:
        ax.plot(epochs, results[key]['val_loss'], color=colors[key], label=labels[key], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Training Accuracy
    ax = axes[0, 2]
    for key in ['ce', 'dice', 'ce_dice']:
        ax.plot(epochs, results[key]['train_acc'], color=colors[key], label=labels[key], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Validation Accuracy
    ax = axes[1, 0]
    for key in ['ce', 'dice', 'ce_dice']:
        ax.plot(epochs, results[key]['val_acc'], color=colors[key], label=labels[key], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Validation mIoU
    ax = axes[1, 1]
    for key in ['ce', 'dice', 'ce_dice']:
        ax.plot(epochs, results[key]['val_miou'], color=colors[key], label=labels[key], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU')
    ax.set_title('Validation mIoU')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Final comparison bar chart
    ax = axes[1, 2]
    best_mious = [max(results[k]['val_miou']) for k in ['ce', 'dice', 'ce_dice']]
    bars = ax.bar([labels[k] for k in ['ce', 'dice', 'ce_dice']], best_mious,
                  color=[colors[k] for k in ['ce', 'dice', 'ce_dice']], alpha=0.8)
    ax.set_ylabel('Best mIoU')
    ax.set_title('Best Validation mIoU Comparison')
    ax.set_ylim(0.4, 0.65)
    for bar, val in zip(bars, best_mious):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
