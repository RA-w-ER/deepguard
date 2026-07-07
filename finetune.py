import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np

sys.path.append(str(Path(__file__).parent))

from src.data.dataset import get_dataloaders
from src.models.detector import DeepfakeDetector



CHECKPOINT = "checkpoints/best_efficientnet_b4.pth"
DATA_DIR   = "data/processed"
SAVE_PATH  = "checkpoints/best_efficientnet_b4.pth"
EPOCHS     = 10
BATCH_SIZE = 32
LR         = 1e-5
NUM_WORKERS = 0



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")


print(f"\nЗагружаем чекпоинт: {CHECKPOINT}")
checkpoint = torch.load(CHECKPOINT, map_location=device)

model = DeepfakeDetector(backbone="efficientnet_b4", pretrained=False)
model.load_state_dict(checkpoint["state_dict"])
model = model.to(device)
print(f"Старый Val AUC: {checkpoint.get('val_auc', '?'):.4f}")


print("\nЗагружаем данные...")
train_loader, val_loader = get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)



optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
criterion = nn.BCEWithLogitsLoss()


best_auc = checkpoint.get("val_auc", 0.0)
print(f"\nСтартуем с AUC: {best_auc:.4f}")
print("=" * 50)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # Train
    model.train()
    total_loss = 0
    all_labels, all_proba = [], []

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_proba.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        if (batch_idx + 1) % 50 == 0:
            print(f"  [{batch_idx+1}/{len(train_loader)}] loss: {total_loss/(batch_idx+1):.4f}", end="\r")

    train_auc = roc_auc_score(all_labels, all_proba)
    train_loss = total_loss / len(train_loader)

    # Val
    model.eval()
    total_loss = 0
    all_labels, all_proba = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_proba.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    val_auc  = roc_auc_score(all_labels, all_proba)
    val_acc  = accuracy_score(all_labels, (np.array(all_proba) >= 0.5).astype(int))
    val_loss = total_loss / len(val_loader)

    scheduler.step()
    elapsed = time.time() - t0

    print(f"Эпоха {epoch:02d}/{EPOCHS} [{elapsed:.0f}с]")
    print(f"  Train — loss: {train_loss:.4f} | AUC: {train_auc:.4f}")
    print(f"  Val   — loss: {val_loss:.4f}  | AUC: {val_auc:.4f} | Acc: {val_acc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save({
            "epoch":      epoch,
            "backbone":   "efficientnet_b4",
            "state_dict": model.state_dict(),
            "val_auc":    val_auc,
            "val_acc":    val_acc,
        }, SAVE_PATH)
        print(f"  ✓ Сохранено (AUC: {best_auc:.4f})")

    print()


print(f"Готово! Лучший Val AUC: {best_auc:.4f}")
