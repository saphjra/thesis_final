# KAAMBA — Thesis Repository

This repository contains the complete codebase and pre-computed results for the thesis project **"Gaze Prediction with Mamba-based Neural Networks"**. It is organised as a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with the KAAMBA package as its sole member.

---

## Repository Structure

```
thesis/
├── kaamba_repo/          # KAAMBA package (model, training, evaluation, inference)
│   └── src/kaamba/
│       ├── net/          # GazePredictor model and image encoders
│       ├── scripts/      # CLI entry points (train, evaluate, infer, stats)
│       └── utils/        # Dataset, loss, preprocessing, evaluation utilities
├── eval_results/         # Pre-computed evaluation results for all four experiments
│   ├── ggtg/             # GGTG dataset · ResNet encoder
│   ├── ggtg_no_encoder/  # GGTG dataset · no image encoder (ablation)
│   ├── mcfw/             # MCFW-Gaze dataset · SigLIP encoder
│   └── mcfw_no_encoder/  # MCFW-Gaze dataset · no image encoder (ablation)
├── data/                 # Downloaded datasets (populated via pymovements, see below)
│   ├── GGTG/
│   └── mcfw-gaze/
└── pyproject.toml        # Workspace configuration
```

Each `eval_results/<experiment>/` folder contains:

| File / Folder | Contents |
|---|---|
| `config.json` | Full training configuration used to produce this model |
| `eval_config.json` | Evaluation parameters (checkpoint path, stimuli, temperature, …) |
| `loader_configs.json` | Train / val / test data-loader splits |
| `final_eval.json` | Per-generator evaluation results |
| `checkpoints/best_model.pt` | Best model checkpoint (stored via Git LFS) |
| `results/aggregate.json` | Aggregated evaluation metrics across all test stimuli |
| `results/metrics.jsonl` | Per-step training metrics |
| `results/eval_report_*.txt` | Human-readable evaluation reports per generator |
| `results/per_stimulus/` | Per-stimulus metric JSON files |
| `visualizations/` | Plots (fixation density, saccade analysis, scanpath overviews, …) |

---

## Requirements

Running this code requires a **Linux machine** with a CUDA-capable GPU. The Mamba-2 backbone relies on [`mamba-ssm`](https://github.com/state-spaces/mamba), which requires compiled Triton kernels and therefore:

- CUDA 12
- Python 3.10 (pinned via `.python-version`)
- A GPU with sufficient VRAM (≥16 GB recommended for training)

[`accelerate`](https://huggingface.co/docs/accelerate) is used for mixed-precision / multi-GPU training. **Windows and macOS are not supported.**

---

## Installation

### 1. Clone with Git LFS

Model checkpoints in `eval_results/*/checkpoints/` are stored in **Git Large File Storage (LFS)**. Install Git LFS before cloning:

```bash
git lfs install
git clone <repo-url>
cd thesis
```

If you already cloned without LFS, run `git lfs pull` to fetch the weight files.

### 2. Install dependencies

This workspace uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv if needed
pip install uv

# Create the virtual environment and install all packages
uv sync
```

The `mamba-ssm` and `causal-conv1d` packages are pre-built CUDA 12 wheels (torch 2.7.1, cp310, linux\_x86\_64) and are fetched automatically from GitHub Releases — no manual wheel installation is needed.

### 3. Activate the environment

```bash
source .venv/bin/activate
```

> **Note on the pymovements version:** this project depends on a **custom development branch** of pymovements (`feat/download_stimulus_files`) that adds stimulus-image downloading support. This branch is declared in `kaamba_repo/pyproject.toml` and installed automatically by `uv sync`. Do **not** substitute the PyPI release — the required APIs are not yet available there.

---

## Downloading the Data

Datasets are downloaded via pymovements. From a Python session or notebook:

```python
import pymovements as pm

# GGTG reading corpus
dataset = pm.Dataset("GGTG", path="data/GGTG")
dataset.download()
dataset.load()

# MCFW-Gaze scene-viewing corpus
dataset = pm.Dataset("mcfw-gaze", path="data/mcfw-gaze")
dataset.download()
dataset.load()
```

Stimulus images are downloaded automatically by the `feat/download_stimulus_files` branch of pymovements. Data is expected under `data/` relative to the repo root (matching the `"root"` field in the eval config files).

### MCFW-Gaze: stimulus preprocessing

The raw MCFW-Gaze stimulus images (`data/mcfw-gaze/raw/dataset/stimuli/`) are smaller than the screen resolution used during the experiment. Before using them for training or inference they must be scaled and placed on a full 1920×1080 canvas with letterbox bars, matching the exact pixel coordinates recorded by the eye-tracker. The functions `scale_image_to_screen` and `place_on_screen` in [`kaamba_repo/src/kaamba/utils/image_preprocessing.py`](kaamba_repo/src/kaamba/utils/image_preprocessing.py) handle this. Run the snippet below once after downloading the dataset:

```python
from pathlib import Path
from PIL import Image
from kaamba.utils.image_preprocessing import scale_image_to_screen, place_on_screen

raw_dir = Path("data/mcfw-gaze/raw/dataset/stimuli")
out_dir = Path("data/mcfw-gaze/stimuli")
out_dir.mkdir(parents=True, exist_ok=True)

screen = (1920, 1080)
for image_id in range(20, 100):
    img = Image.open(raw_dir / f"{image_id}.jpg")
    scaled_img, offset_x, offset_y, _, _ = scale_image_to_screen(img, screen)
    screen_img = place_on_screen(scaled_img, screen, offset_x, offset_y)
    screen_img.save(out_dir / f"{image_id}.jpg")
```

The processed images are written to `data/mcfw-gaze/stimuli/`, which is where the dataset pipeline expects to find them.

---

## Reproducing the Results

Each experiment has a self-contained `eval_config.json` that records the checkpoint path, dataset, stimuli, and all preprocessing parameters. Pass it to `evaluate_model` with the `--config` flag and assign a label for the output directory:

```bash
# GGTG · ResNet encoder
python -m kaamba.scripts.evaluate_model \
    --config eval_results/ggtg/eval_config.json \
    model --label ggtg

# GGTG · no encoder (ablation)
python -m kaamba.scripts.evaluate_model \
    --config eval_results/ggtg_no_encoder/eval_config.json \
    model --label ggtg_noe

# MCFW-Gaze · SigLIP encoder
python -m kaamba.scripts.evaluate_model \
    --config eval_results/mcfw/eval_config.json \
    model --label mcfw

# MCFW-Gaze · no encoder (ablation)
python -m kaamba.scripts.evaluate_model \
    --config eval_results/mcfw_no_encoder/eval_config.json \
    model --label mcfw_noe
```

Results are written to the directory specified by `--out_dir` (defaults to the config's own folder). Each run produces per-stimulus JSON files, an `aggregate.json`, and an `eval_report.txt`.

> **Paths in eval_config.json:** the `"root"` field is set to `/home/janhof/thesis/data` (the original training server). Adjust this to your local data path if you are reproducing on a different machine.

---

## Training from Scratch

See [`kaamba_repo/README.md`](kaamba_repo/README.md) for the full training guide, including encoder options, Optuna hyperparameter search, and inference instructions.

---

## License

MIT
