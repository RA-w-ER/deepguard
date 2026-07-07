import cv2
from pathlib import Path
from tqdm import tqdm

SRC = Path(r"C:\Users\RAWER\Downloads\real_vs_fake\real-vs-fake")
DST = Path(r"C:\Users\RAWER\PycharmProjects\deepfake-detector\data\processed")
MAX_PER_CLASS = 30000

for label in ["real", "fake"]:
    dst = DST / label
    dst.mkdir(parents=True, exist_ok=True)
    old = list(dst.glob("gan_*.png"))
    print(f"[{label.upper()}] Удаляем {len(old)}")
    for f in old:
        f.unlink()
    images = []
    for split in ["train", "valid", "test"]:
        folder = SRC / split / label
        if folder.exists():
            images += list(folder.glob("*.jpg"))

    images = images[:MAX_PER_CLASS]
    print(f"[{label.upper()}] Копируем {len(images)}")

    for img_path in tqdm(images, desc=label):
        dst_path = dst / f"gan_{img_path.stem}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(str(dst_path), img)

print("\nГ!")
for label in ["real", "fake"]:
    total = len(list((DST / label).glob("*.png")))
    gan   = len(list((DST / label).glob("gan_*.png")))
    ff    = total - gan
    print(f"  {label}: {total} всего | FF++: {ff} | GAN: {gan}")