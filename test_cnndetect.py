

import sys
import subprocess
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T



WEIGHTS_PATH = Path("checkpoints/cnndetect_resnet50.pth")

if not WEIGHTS_PATH.exists():
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Скачиваем веса CNNDetect...")
    subprocess.run([
        sys.executable, "-m", "gdown",
        "https://drive.google.com/uc?id=1ni_HSPtMn6An0LyrGIi5HuWBKXN1JFzA",
        "-O", str(WEIGHTS_PATH)
    ])
    print("Готово!\n")
else:
    print(f"Веса найдены: {WEIGHTS_PATH}\n")


model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)

state = torch.load(WEIGHTS_PATH, map_location="cpu")

if "model" in state:
    model.load_state_dict(state["model"])
elif "state_dict" in state:
    model.load_state_dict(state["state_dict"])
else:
    model.load_state_dict(state)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"CNNDetect загружен | Устройство: {device}\n")



transform = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


TEST_DIR = Path("phototest")
image_paths = sorted(
    list(TEST_DIR.glob("*.jpg")) +
    list(TEST_DIR.glob("*.jpeg")) +
    list(TEST_DIR.glob("*.png"))
)

print(f"Анализируем {len(image_paths)} \n")
print("─" * 50)

results = []
for image_path in image_paths:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ⚠ [{image_path.name}] Ошибка чтения")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    side    = min(h, w)
    top     = (h - side) // 2
    left    = (w - side) // 2
    img_rgb = img_rgb[top:top+side, left:left+side]
    img_rgb = cv2.resize(img_rgb, (256, 256))

    inp = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        proba = torch.sigmoid(model(inp)).item()

    label      = "FAKE" if proba >= 0.5 else "REAL"
    confidence = proba if label == "FAKE" else 1.0 - proba
    bar        = "█" * int(confidence * 100 / 5) + "░" * (20 - int(confidence * 100 / 5))
    icon       = "🔴" if label == "FAKE" else "🟢"

    print(f"  {icon} [{image_path.name}]")
    print(f"     Результат:   {label}")
    print(f"     Уверенность: {confidence * 100:.1f}%  |{bar}|")
    print(f"     P(fake):     {proba:.4f}")

    results.append({"label": label})

n_fake = sum(1 for r in results if r["label"] == "FAKE")
n_real = sum(1 for r in results if r["label"] == "REAL")
print("\n" + "─" * 50)
print(f"ИТОГО: {len(results)} файлов")
print(f"  🟢 REAL: {n_real}")
print(f"  🔴 FAKE: {n_fake}")