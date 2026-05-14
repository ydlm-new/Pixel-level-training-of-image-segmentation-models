import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch
import random


class StanfordBackgroundDataset(Dataset):
    def __init__(self, image_dir, label_dir, file_list, img_size=(256, 256), augment=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.file_list = file_list
        self.img_size = img_size
        self.augment = augment
        self.num_classes = 8

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        base_name = img_name.replace('.jpg', '')

        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, base_name + '.regions.txt')

        image = Image.open(img_path).convert('RGB')
        label = np.loadtxt(label_path).astype(np.int64)

        # Map negative labels (unknown) to 255 (ignore index)
        label[label < 0] = 255

        label = Image.fromarray(label.astype(np.uint8))

        # Resize
        image = image.resize(self.img_size, Image.BILINEAR)
        label = label.resize(self.img_size, Image.NEAREST)

        # Data augmentation
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                label = TF.rotate(label, angle, interpolation=TF.InterpolationMode.NEAREST)

        # To tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        label = torch.from_numpy(np.array(label)).long()

        return image, label


def get_file_lists(image_dir, train_ratio=0.8, seed=42):
    all_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    random.seed(seed)
    random.shuffle(all_files)
    split = int(len(all_files) * train_ratio)
    train_files = all_files[:split]
    val_files = all_files[split:]
    return train_files, val_files
