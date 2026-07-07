import cv2
from pathlib import Path

for f in Path("phototest").glob("*.jpg"):
    img = cv2.imread(str(f))
    print(f"{f.name}: {img.shape}")