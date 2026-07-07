import cv2
from pathlib import Path
from tqdm import tqdm

SRC = Path(r"C:\Users\RAWER\Downloads\stylegan3_faces\StyleGAN3_256x256")
DST = Path(r"C:\Users\RAWER\PycharmProjects\deepfake-detector\data\processed\fake")

DST.mkdir(parents=True, exist_ok=True)

images = list(SRC.glob("*.png"))
print(f"Найдено {len(images)} StyleGAN3 изображений")

saved = 0
for img_path in tqdm(images, desc="stylegan3"):
    dst_path = DST / f"sg3_{img_path.stem.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    if dst_path.exists():
        continue
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    img = cv2.resize(img, (256, 256))
    cv2.imwrite(str(dst_path), img)
    saved += 1

print(f"\nДобавлено {saved} новых fake-изображений (StyleGAN3)")

total = len(list(DST.glob("*.png")))
sg2   = len(list(DST.glob("gan_*.png")))
sg3   = len(list(DST.glob("sg3_*.png")))
print(f"\nВсего в fake/: {total}")
print(f"  StyleGAN2: {sg2}")
print(f"  StyleGAN3: {sg3}")
