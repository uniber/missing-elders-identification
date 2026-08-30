# Missing Elders Identification — Automatic Lost-Elderly Detection

A real-time system that automatically detects likely lost elderly people from surveillance-camera images. It receives images over HTTP, detects each person, recognizes whether they are elderly and extracts their attributes with a CLIP-based cross-modal model (built on [IRRA](https://github.com/anosky/RSTP-Reid)), tracks them across frames, and reasons over multiple cues with probabilistic logic (ProbLog) to estimate a *lost* probability.

## Pipeline

```
camera image ──> serve.py (FastAPI) ──> img_receive folder
                                              │
                        finally.py polls for new images
                                              │
              ┌───────────────────────────────┘
              ▼
   1. Person detection        YOLO (yolo11s.pt)
   2. Elderly recognition     CLIP/IRRA text-image matching ("elderly person" vs "young person")
   3. Attribute extraction    clothing / behavior / posture cues via text-image matching
   4. Re-identification       feature similarity against previously seen elders
   5. Probabilistic reasoning ProbLog combines speed, lingering time, night, alone → lost probability
                                              │
                              lost_p > 0.5 ──> flag & save to save_files/lost/
```

## Core Files

| File | Description |
| --- | --- |
| `finally.py` | Main program: YOLO person detection → elderly/attribute recognition → tracking → ProbLog lost-risk reasoning |
| `serve.py` | FastAPI server that receives camera images and saves them to the watched folder |
| `receive_img.py` | Video frame extraction (video → image sequence at a given frame rate) |
| `train.py` | Training entry point for the IRRA (CLIP-based) backbone |
| `run_this.py` | Interactive inference demo (text + image path → similarity) |

## How the detection works

1. **Person detection** — `check_image_information()` runs YOLO (`yolo11s.pt`) and crops every detected person (`class 0`), encoding each with the IRRA/CLIP model.
2. **Elderly recognition** — `check_if_elder()` compares each person's feature against the text embeddings `"elderly person"` / `"young person"` (softmax, threshold `0.70`).
3. **Attribute classification** — the model also scores a set of curated text descriptions (medical gown, wristband, backpack, clothing, posture such as sitting/walking/standing, abnormal behavior, night) against each person image.
4. **Re-identification & tracking** — `check_if_appear()` matches the person against a running list of previously seen elders, tracking location, time, and speed across frames.
5. **Lost-risk reasoning** — `calculate_lost()` writes a ProbLog program (`lost.pl`) with probabilistic rules over `time` (night), `weather`, `stay` (lingering), and `slow` (walking speed), then queries `e` to get the lost probability. If it exceeds `0.5`, the person is flagged and the image is saved to `save_files/lost/`.

## Directory Structure

```
├── finally.py            # Main program (detection + reasoning)
├── serve.py              # FastAPI server
├── receive_img.py        # Video frame extraction
├── train.py              # Training entry point
├── run_this.py           # Inference demo
├── datasets/             # Data loading / samplers / preprocessing
├── model/                # CLIP model and objectives
├── processor/            # Training / evaluation loops
├── solver/               # Optimizer and learning-rate scheduler
├── utils/                # Config, logger, tokenizer, checkpoint, metrics
├── data/
│   └── bpe_simple_vocab_16e6.txt.gz   # CLIP BPE vocabulary (required by tokenizer)
├── logs/
│   └── loss_plt.py       # Loss visualization
└── yolo11s.pt            # YOLOv11s weights
```

## Dependencies

- Python 3.x, PyTorch (CUDA)
- `ultralytics` (YOLO)
- `problog` (probabilistic logic reasoning)
- `fastapi`, `uvicorn` (server)
- `torchvision`, `opencv-python`, `matplotlib`, `PIL`, `ftfy`, `regex`, etc.

## Training

The IRRA (CLIP-based) backbone is trained on RSTPReid:

```bash
python train.py --name baseline --dataset_name RSTPReid
```

Training outputs are written to `logs/RSTPReid/<timestamp>_<name>/`, including the `best.pth` model weights and `configs.yaml`.

## Inference

```bash
# Main program (automatic lost-elderly detection)
python finally.py

# FastAPI server that feeds images into the watched folder
python serve.py
```

`finally.py` loads the trained `configs.yaml` and `best.pth`; their paths are set via the `config_file` argument of `run_model()`.

## Model Weights and Dataset

The following large files are **not** included in this repository (they exceed GitHub's 100 MB per-file limit). Obtain them locally:

- **`best.pth`** (~1.3 GB): trained IRRA model weights at `logs/RSTPReid/<run>/best.pth`, produced by `train.py`.
- **RSTPReid dataset** (~500 MB): person description-image dataset, place under `data/RSTPReid/` (contains `imgs/` and `data_captions.json`).
- `yolo11s.pt` can also be auto-downloaded by `ultralytics` on first run.

## Notes

Some paths in `serve.py` and `receive_img.py` (e.g. `E:\python_projects\img_receive`, the video path) are hard-coded locally; adjust them as needed.
