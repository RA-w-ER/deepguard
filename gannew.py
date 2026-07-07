from pathlib import Path

for label in ["real", "fake"]:
    folder = Path(r"C:\Users\RAWER\PycharmProjects\deepfake-detector\data\processed") / label
    for f in folder.glob("gan_*.png"):
        f.rename(f.parent / f.name.replace("gan_", ""))

print("Готово!")
for label in ["real", "fake"]:
    count = len(list((Path(r"C:\Users\RAWER\PycharmProjects\deepfake-detector\data\processed") / label).glob("*.png")))
    print(f"  {label}: {count} файлов")