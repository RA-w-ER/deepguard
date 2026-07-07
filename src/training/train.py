import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.dataset import get_dataloaders
from src.models.detector import DeepfakeDetector



EPOCHS= 15
BATCH_SIZE= 32
LR= 1e-4
WEIGHT_DECAY= 1e-4
NUM_WORKERS= 0
DATA_DIR= "data/processed"
SAVE_DIR= Path("checkpoints")
SAVE_NAME= "best_resnet50.pth"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda rtx" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*50}")
print(f"Устройство:  {device}")
print(f"Backbone:    ResNet50")
print(f"Эпох:        {EPOCHS}")
print(f"Batch size:  {BATCH_SIZE}")
print(f"LR:          {LR}")
print(f"{'='*50}\n")


print("данные...")
train_loader, val_loader = get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)


print("\n модель...")
model = DeepfakeDetector(pretrained=True, dropout=0.5).to(device)

optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = nn.BCEWithLogitsLoss()

best_val_auc = 0.0
best_epoch   = 0

print("\n обучение...\n")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    total_loss, all_labels, all_proba = 0.0, [], []

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_proba.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  [{batch_idx+1}/{len(train_loader)}] "
                  f"loss: {total_loss/(batch_idx+1):.4f}", end="\r")

    print()
    train_preds = (np.array(all_proba) >= 0.5).astype(int)
    train_loss  = total_loss / len(train_loader)
    train_auc   = roc_auc_score(all_labels, all_proba)
    train_acc   = accuracy_score(all_labels, train_preds)
    train_f1    = f1_score(all_labels, train_preds, zero_division=0)

    model.eval()
    total_loss, all_labels, all_proba = 0.0, [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            total_loss += criterion(logits, labels).item()
            all_proba.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    val_preds = (np.array(all_proba) >= 0.5).astype(int)
    val_loss  = total_loss / len(val_loader)
    val_auc   = roc_auc_score(all_labels, all_proba)
    val_acc   = accuracy_score(all_labels, val_preds)
    val_f1    = f1_score(all_labels, val_preds, zero_division=0)

    scheduler.step()

    print(f"Эпоха {epoch:02d}/{EPOCHS} [{time.time()-t0:.0f}с]")
    print(f"  Train  loss: {train_loss:.4f} | AUC {train_auc:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"  Val    loss: {val_loss:.4f} | AUC {val_auc:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch   = epoch
        torch.save({
            "epoch":      epoch,
            "backbone":   "resnet50",
            "state_dict": model.state_dict(),
            "val_auc":    val_auc,
            "val_acc":    val_acc,
        }, SAVE_DIR / SAVE_NAME)
        print(f"Сохранено (AUC {best_val_auc:.4f})")

    print()
print(f"Лучший Val AUC {best_val_auc:.4f} (эпоха {best_epoch})")
print(f"Модель checkpoints/{SAVE_NAME}")

