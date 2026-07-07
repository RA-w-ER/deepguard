from pathlib import Path
src = Path(r"C:\Users\RAWER\Downloads\real_vs_fake\real-vs-fake")
for split in ["train", "valid", "test"]:
    for label in ["real", "fake"]:
        count = len(list((src / split / label).glob("*.jpg")))
        print(f"{split}/{label}: {count}")