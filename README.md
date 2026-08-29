# IRRA Text-to-Image Person Retrieval

A text-to-image person re-identification (ReID) project built on CLIP, based on IRRA (Implicit Relation Reasoning and Aligning). It integrates YOLO person detection to deliver a full pipeline: camera stream → person detection → text-description retrieval.

## Core Files

| File | Description |
| --- | --- |
| `finally.py` | Main program: YOLO person detection + IRRA text-image retrieval |
| `serve.py` | FastAPI server that receives camera images and saves them |
| `receive_img.py` | Video frame extraction (video → image sequence at a given frame rate) |
| `train.py` | Training entry point |
| `run_this.py` | Interactive inference demo (text + image path → similarity) |

## Directory Structure

```
├── finally.py            # Main program (YOLO + IRRA)
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
- `fastapi`, `uvicorn` (server)
- `torchvision`, `opencv-python`, `matplotlib`, `PIL`, `ftfy`, `regex`, etc.

## Training

```bash
python train.py --name baseline --dataset_name RSTPReid
```

Training outputs are written to `logs/RSTPReid/<timestamp>_<name>/`, including the `best.pth` model weights and `configs.yaml`.

## Inference

```bash
# Interactive demo
python run_this.py

# Main program (YOLO detection + retrieval)
python finally.py
```

`finally.py` loads the trained `configs.yaml` and `best.pth`; their paths are set via the `config_file` argument of `run_model()`.

## Model Weights and Dataset

The following large files are **not** included in this repository (they exceed GitHub's 100 MB per-file limit). Obtain them locally:

- **`best.pth`** (~1.3 GB): trained IRRA model weights at `logs/RSTPReid/<run>/best.pth`, produced by `train.py`.
- **RSTPReid dataset** (~500 MB): person description-image dataset, place under `data/RSTPReid/` (contains `imgs/` and `data_captions.json`).
- `yolo11s.pt` can also be auto-downloaded by `ultralytics` on first run.

## Notes

Some paths in `serve.py` and `receive_img.py` (e.g. `E:\python_projects\img_receive`, the video path) are hard-coded locally; adjust them as needed.
