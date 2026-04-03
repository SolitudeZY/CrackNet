import torch
import torch.nn as nn
import torch.nn.functional as F
from model import U2NETP
from train import build_deepcrack_datasets, CONFIG

cfg = CONFIG.copy()
cfg["batch_size"] = 2
cfg["num_workers"] = 0

train_loader, val_loaders = build_deepcrack_datasets(cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = U2NETP(3, 1).to(device)
model.train()

scaler = torch.amp.GradScaler("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for i, (images, masks) in enumerate(train_loader):
    images, masks = images.to(device), masks.to(device)
    optimizer.zero_grad()
    
    with torch.amp.autocast("cuda"):
        outputs = model(images)
        d0 = outputs[0]
        loss = F.binary_cross_entropy_with_logits(d0, masks)
        
    print(f"loss: {loss.item()}")
    
    scaler.scale(loss).backward()
    
    scaler.unscale_(optimizer)
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    print(f"grad_norm: {grad_norm}")
    
    scaler.step(optimizer)
    scaler.update()
    
    # print some weights to see if they are updating normally
    for name, param in model.named_parameters():
        if "conv_s1.weight" in name:
            print(f"{name}: {param.mean().item()}")
            break
            
    if i > 5:
        break
