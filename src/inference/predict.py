import sys
import argparse
from pathlib import Path

import cv2
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.detector import DeepfakeDetector


parser = argparse.ArgumentParser()
parser.add_argument("--input",      required=True)
parser.add_argument("--checkpoint", default="checkpoints/best_resnet50.pth")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}\n")



checkpoint = torch.load(args.checkpoint, map_location=device)
model = DeepfakeDetector(pretrained=False)
model.load_state_dict(checkpoint["state_dict"])
model.eval().to(device)
print(f"Модель загружена: {checkpoint.get('backbone', 'resnet50')}")
print(f"Val AUC при сохранении: {checkpoint.get('val_auc', 0):.4f}\n")

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



input_path = Path(args.input)
if input_path.is_file():
    image_paths = [input_path]
else:
    image_paths = sorted(
        list(input_path.glob("*.jpg")) +
        list(input_path.glob("*.jpeg")) +
        list(input_path.glob("*.png"))
    )

print(f"Анализируем {len(image_paths)} изображений\n")



results = []
for image_path in image_paths:
    img = Image.open(str(image_path)).convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        proba = torch.sigmoid(model(inp)).item()

    label      = "FAKE" if proba >= 0.5 else "REAL"
    confidence = proba if label == "FAKE" else 1.0 - proba
    bar        = "█" * int(confidence * 100 / 5) + " " * (20 - int(confidence * 100 / 5))
    icon       = "X" if label == "FAKE" else "V"

    print(f"  {icon} [{image_path.name}]")
    print(f"     Результат:   {label}")
    print(f"     Уверенность: {confidence * 100:.1f}%  |{bar}|")
    print(f"     P(fake):     {proba:.4f}")

    results.append({"label": label})



if len(results) > 1:
    n_fake = sum(1 for r in results if r["label"] == "FAKE")
    n_real = sum(1 for r in results if r["label"] == "REAL")
    print(f"ИТОГО: {len(results)}")
    print(f"    REAL: {n_real}")
    print(f"    FAKE: {n_fake}")
