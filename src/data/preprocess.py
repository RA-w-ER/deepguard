
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN




FACE_SIZE = 224
FRAMES_PER_VIDEO = 30
MARGIN = 30
MIN_FACE_SIZE = 80
CONFIDENCE_THRESHOLD = 0.9

FF_PATHS = {
    "real": "original_sequences/youtube/c23/videos",
    "fake": "manipulated_sequences/Deepfakes/c23/videos",
}




def get_detector(device: str) -> MTCNN:
    return MTCNN(
        image_size=FACE_SIZE,
        margin=MARGIN,
        min_face_size=MIN_FACE_SIZE,
        thresholds=[0.6, 0.7, 0.9],
        factor=0.709,
        post_process=False,
        device=device,
        keep_all=False,
    )




def extract_faces_from_video(
    video_path: Path,
    output_dir: Path,
    detector: MTCNN,
    frames_per_video: int = FRAMES_PER_VIDEO,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Не удалось открыть: {video_path.name}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return 0

    frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)
    frame_indices = np.unique(frame_indices)

    saved = 0
    video_id = video_path.stem

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            face_tensor, prob = detector(frame_rgb, return_prob=True)
        except Exception:
            continue

        if face_tensor is None or prob is None:
            continue
        if prob < CONFIDENCE_THRESHOLD:
            continue

        face_np = face_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

        out_path = output_dir / f"{video_id}_{frame_idx:04d}.png"
        cv2.imwrite(str(out_path), face_bgr)
        saved += 1

    cap.release()
    return saved


def process_category(
    raw_dir: Path,
    output_dir: Path,
    category: str,
    detector: MTCNN,
    max_videos: int = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = raw_dir / FF_PATHS[category]

    if not video_dir.exists():
        print(f"  [ERROR] Папка не найдена: {video_dir}")
        return {"category": category, "videos_processed": 0,
                "videos_failed": 0, "faces_saved": 0}

    all_videos = sorted(video_dir.glob("*.mp4"))
    print(f"  Найдено {len(all_videos)} видео в {FF_PATHS[category]}")

    if max_videos:
        all_videos = all_videos[:max_videos]
        print(f"  Ограничение: берём первые {max_videos}")

    total_saved = 0
    failed = 0

    for video_path in tqdm(all_videos, desc=f"[{category}]", unit="video"):
        n = extract_faces_from_video(video_path, output_dir, detector)
        if n == 0:
            failed += 1
        total_saved += n

    return {
        "category": category,
        "videos_processed": len(all_videos) - failed,
        "videos_failed": failed,
        "faces_saved": total_saved,
    }


parser = argparse.ArgumentParser(
    description="Предобработка FF++ — извлечение лиц из видео"
)
parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
parser.add_argument("--frames-per-video", type=int, default=FRAMES_PER_VIDEO)
parser.add_argument(
    "--max-videos", type=int, default=None,
    help="Ограничить число видео (для теста: --max-videos 10)"
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Устройство:         {device}")
print(f"Кадров из видео:    {args.frames_per_video}")
print(f"Размер кропа:       {FACE_SIZE}x{FACE_SIZE}")
if args.max_videos:
    print(f"Ограничение:        {args.max_videos} в")
print()

print("Инициализация MTCNN детектора лиц")
detector = get_detector(device)
print("OK\n")

all_stats = []
for category in ["real", "fake"]:
    print(f"── Категория: {category.upper()} ──")
    stats = process_category(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir / category,
        category=category,
        detector=detector,
        max_videos=args.max_videos,
    )
    all_stats.append(stats)
    print()


print("ИТОГ:")
for s in all_stats:
    print(f"\n  [{s['category'].upper()}]")
    print(f"    Видео обработано:  {s['videos_processed']}")
    print(f"    Видео с ошибкой:   {s['videos_failed']}")
    print(f"    Лиц сохранено:     {s['faces_saved']}")

total = sum(s["faces_saved"] for s in all_stats)
print(f"\n  ВСЕГО изображений: {total}")
print("=" * 45)
print("\nГотово! Следующий шаг: python src/data/dataset.py")



