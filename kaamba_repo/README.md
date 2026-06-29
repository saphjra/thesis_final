# KAAMBA: Gaze Prediction with Mamba-based Neural Networks
This repository accompanies a master's thesis and contains the full training, evaluation, and inference pipeline.

---

## Repository Structure

```
src/kaamba/
├── net/
│   └── models/kaamba.py          # GazePredictor model definition
├── scripts/
│   ├── unified_training.py       # Training loop with optional Optuna HPO
│   ├── evaluate_model.py         # Evaluation against baselines
│   ├── infer.py                  # Single-image scanpath inference
│   └── dataset_stats.py          # Dataset statistics and plots
└── utils/
    ├── on_the_fly_dataset.py     # Streaming dataset (pymovements interface)
    ├── loss_functions.py         # GMM NLL loss
    ├── gaze_preprocessing.py     # Fixation/saccade event detection
    └── ...
```

---

## Requirements

### Hardware

Running this code requires a Linux machine with a CUDA-capable GPU. The Mamba-2 sequence model depends on [mamba-ssm](https://github.com/state-spaces/mamba), which requires:

- CUDA 12
- Python 3.10
- A GPU with sufficient VRAM for training (≥16 GB recommended)

[`accelerate`](https://huggingface.co/docs/accelerate) is used for distributed/mixed-precision training, and the `mamba-ssm` and `causal-conv1d` packages use [Triton](https://github.com/triton-lang/triton) kernels under the hood. **Windows and macOS are not supported.**

---

## Installation

### 1. Clone the repository (with Git LFS)

Pre-trained model weights are stored in Git Large File Storage (LFS). You must have [Git LFS](https://git-lfs.com/) installed before cloning, otherwise the weight files will be placeholders.

```bash
# Install Git LFS (once per machine)
git lfs install

# Clone the repository
git clone <repo-url>
cd kaamba_repo
```

If you already cloned without LFS, run `git lfs pull` inside the repo to fetch the weights.

### 2. Set up the Python environment

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and is pinned to **Python 3.10**.

```bash
# Install uv if needed
pip install uv

# Create a virtual environment and install all dependencies
uv sync
```

The `mamba-ssm` and `causal-conv1d` packages are pre-built CUDA 12 wheels (for `torch==2.7.1`, `cp310`, `linux_x86_64`) and are fetched automatically from GitHub Releases during `uv sync`. No manual wheel installation is needed.

### 3. Activate the environment

```bash
source .venv/bin/activate
```

---

## A Note on the pymovements Dependency

This project uses a **custom development branch** of [pymovements](https://github.com/pymovements/pymovements):

```
branch: feat/download_stimulus_files
```

This branch adds support for automatic stimulus image downloading, which is required for the dataset pipeline. The branch is tracked in `pyproject.toml` and installed automatically by `uv sync`. Do **not** substitute the PyPI release of `pymovements` — the required APIs are not yet available there.

---

## Downloading the Data

Gaze datasets are downloaded via pymovements. The following datasets are supported and can be loaded by name:

- `GGTG`
- `mcfw-gaze`
- *(others as configured in your experiment)*

To download a dataset, use the pymovements download interface. For example, in a Python session or notebook:

```python
import pymovements as pm

dataset = pm.Dataset("GGTG", path="data/GGTG")
dataset.download()
dataset.load()
```

Stimulus images are downloaded automatically when using the `feat/download_stimulus_files` branch. Data is expected under a `data/` directory at the repo root (or as configured in your training script).

---

## Usage

### Training

```bash
python -m kaamba.scripts.unified_training --datasets GGTG mcfw-gaze --encoder siglip --conditioning every_step --log_dir runs
```

For hyperparameter search with Optuna, pass `--hparam_search` along with a study name. Run `python -m kaamba.scripts.unified_training --help` for all options.

### Evaluation

```bash
python -m kaamba.scripts.evaluate_model --checkpoint runs/<run_id>/checkpoints/best_model.pt --datasets GGTG --out_dir eval_results/
```

This evaluates the model against synthetic and empirical baselines and writes per-stimulus and aggregate metric reports.

### Inference

```bash
python -m kaamba.scripts.infer --checkpoint runs/<run_id>/checkpoints/best_model.pt --image path/to/stimulus.png --n 1
```

Generates 1 scanpath samples overlaid on the stimulus image.

---

## License

MIT
