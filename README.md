# Детектор дипфейков — кросс-генераторная детекция синтетических лиц

Система автоматического обнаружения изображений лиц, сгенерированных нейросетями (GAN). Проект разрабатывается в рамках конкурса **ITMO STARS 2026**, направление «Информационная безопасность».

## Проблема

Большинство детекторов дипфейков отлично работают на изображениях того генератора, на котором обучались, но полностью теряют точность на изображениях из **другого** генератора того же класса (StyleGAN2 → StyleGAN3, ProGAN и т.д.). Это явление называется **domain shift** и является одной из ключевых нерешённых проблем в области deepfake-детекции.

## Архитектура

- **Backbone:** ResNet50 (предобучен на ImageNet)
- **Методология обучения:** CNNDetect (Wang et al., CVPR 2020) — агрессивная аугментация Gaussian Blur + JPEG-компрессией переменного качества
- **Данные:** объединение нескольких генеративных источников — StyleGAN2, StyleGAN3, ProGAN (диверсификация — ключевой фактор обобщающей способности модели)

## Результаты

| Подход | Backbone | Данные | Val AUC | Точность на новых данных |
|---|---|---|---|---|
| Baseline RGB | EfficientNet-B4 | StyleGAN2 | 1.0000 | 28.6% |
| RGB + FFT | EfficientNet-B4 | StyleGAN2 | 1.0000 | 33.3% |
| RGB + SRM | EfficientNet-B4 | StyleGAN2 | 0.9999 | 33.3% |
| CNNDetect | ResNet50 | StyleGAN2 | 0.9999 | 22.2% |
| **CNNDetect + diverse** | **ResNet50** | **StyleGAN2+3** | **0.9999** | **80%** |

Полный анализ, графики и методология — в [подробном отчёте](report.pdf).

## Структура проекта

```
deepfake-detector/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset и DataLoader, аугментации
│   │   └── preprocess.py       # Предобработка исходных изображений
│   ├── inference/
│   │   └── predict.py          # Инференс на изображениях
│   ├── models/
│   │   ├── __init__.py
│   │   └── detector.py         # Архитектура модели
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py            # Скрипт обучения
│   └── __init__.py
|
├── prepare_kaggle_faces.py # Подготовка датасета 140k Real and Fake Faces
├── add_stylegan3.py        # Добавление StyleGAN3-изображений
├── add_progan.py           # Добавление ProGAN-изображений
├── report.docx                 # Подробный отчёт с графиками
├── phototest/                  # Примеры тестовых изображений
└── README.md
```

## Датасет и веса модели

Из-за размера датасета (десятки тысяч изображений) и весов модели (сотни МБ) они не хранятся в этом репозитории. Всё доступно по ссылке на Google Диск:

**Google Drive:** [ссылка на checkpoint модели](https://drive.google.com/drive/folders/1jZ-EY_VNeIz2jpUR-2HcsqXVRKs0uxx7?usp=sharing)

Папка содержит:
- best_resnet50.pth — обученная модель
- Датасет слишком большой и не поммещается на диске

## Установка и запуск

```bash
git clone https://github.com/RA-w-ER/deepfake-detector.git
cd deepfake-detector

python -m venv .venv
.venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install timm albumentations facenet-pytorch scikit-learn opencv-python tqdm
```

Скачай датасет и веса модели по ссылке выше, помести:
- датасет → `data/processed/real/` и `data/processed/fake/`
- веса → `checkpoints/best_resnet50.pth`

### Обучение модели

```bash
python src/training/train.py
```

### Тест на изображении

```bash
python src/inference/predict.py --input path/to/photo.jpg
```

## Технологический стек

- Python, PyTorch (CUDA 12.8)
- timm, torchvision — архитектуры моделей
- albumentations — аугментации
- facenet-pytorch (MTCNN) — детекция лиц
- OpenCV — обработка изображений, частотный анализ
- scikit-learn — метрики (AUC-ROC, F1, Accuracy)

## Дальнейшее развитие

- Добавление диффузионных моделей (Stable Diffusion) и face-swap дипфейков
- Grad-CAM визуализация для интерпретируемости решений
- Ансамблирование RGB / частотного / face-consistency детекторов
- Веб-интерфейс на FastAPI
- Расширение на видео и аудио-визуальную согласованность

## Автор

Семён — ITMO STARS 2026, направление «Информационная безопасность»
