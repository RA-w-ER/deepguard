import cv2
from pathlib import Path
from tqdm import tqdm

SRC = Path(r"C:\Users\RAWER\Downloads\stylegan3_faces\ProGAN_128x128")
DST = Path(r"C:\Users\RAWER\PycharmProjects\deepfake-detector\data\processed\fake")

DST.mkdir(parents=True, exist_ok=True)

images = list(SRC.glob("*.png"))
print(f"Найдено {len(images)} ProGAN изображений (128x128)")

saved = 0
for img_path in tqdm(images, desc="progan"):
    dst_path = DST / f"progan_{img_path.stem.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    if dst_path.exists():
        continue
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    # апскейл 128x128 -> 256x256, чтобы совпадало с остальными данными
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(dst_path), img)
    saved += 1

print(f"\nДобавлено {saved} новых fake-изображений (ProGAN)")

total  = len(list(DST.glob("*.png")))
sg3    = len(list(DST.glob("sg3_*.png")))
progan = len(list(DST.glob("progan_*.png")))
sg2    = total - sg3 - progan
print(f"\nВсего в fake/: {total}")
print(f"  StyleGAN2: {sg2}")
print(f"  StyleGAN3: {sg3}")
print(f"  ProGAN:    {progan}")
