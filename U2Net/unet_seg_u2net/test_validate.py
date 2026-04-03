import torch
from train import build_deepcrack_datasets, CONFIG
from train import validate, calculate_metrics
import numpy as np

cfg = CONFIG.copy()
cfg["batch_size"] = 2
cfg["num_workers"] = 0

train_loader, val_loaders = build_deepcrack_datasets(cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_on_loader(name, loader):
    print(f"\n{name}")
    total_samples = 0
    zero_mask_samples = 0
    for images, masks in loader:
        for i in range(masks.shape[0]):
            total_samples += 1
            if masks[i].sum().item() == 0:
                zero_mask_samples += 1
    print(f"Total samples: {total_samples}, Zero mask samples: {zero_mask_samples}, Ratio: {zero_mask_samples/total_samples:.4f}")

for name, loader in val_loaders.items():
    test_on_loader(name, loader)

