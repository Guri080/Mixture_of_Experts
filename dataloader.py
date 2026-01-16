import os
import torch
import random
import copy
import csv
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
import numpy as np

import medmnist
from medmnist import INFO, Evaluator

class MedMNIST_2D_Datasets(Dataset):
    def __init__(self, dataset_name, split, download=True, as_rgb=True, size=28, root='/scratch/gssodhi/', augment=None):

        self.name = dataset_name
        # get dataset info
        info = INFO[str(dataset_name)]
        DataClass = getattr(medmnist, info['python_class'])
        self.augment = augment
        self.num_classes = len(info['label'])

        self.task = info['task']

        # instantiate the medmnist dataset
        self.dataset = DataClass(
            split=split,
            transform=augment,
            download=download,
            as_rgb=as_rgb,
            size=size,
            root=root
        )
        
        if split == 'train':
            # Training: WITH augmentation
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),  # Resize first
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                ),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            # Val/Test: NO augmentation
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __getitem__(self, index):
        # MedMNIST returns (image, target) already
        img, label = self.dataset[index]
        
        img = self.augment(train_augment)

        # --- normalize labels for BCE loss ---
        if self.task == "multi-label":
            # Already multi-hot (e.g. ChestMNIST) → just cast to float
            label = torch.tensor(label, dtype=torch.float32)
        elif self.task in ["multi-class", "binary-class", "ordinal-regression"]:
            # Convert single integer class into one-hot
            if hasattr(label, 'ndim') and label.ndim > 0:  
                label = int(label.item())
            one_hot = torch.zeros(self.num_classes, dtype=torch.float32)
            one_hot[label] = 1.0
            label = one_hot
                
        return img, label

        def __len__(self):
            return len(self.dataset)