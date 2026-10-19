# Adversarial Attack Script Generator

Interactive tool that generates a ready-to-run adversarial attack script.

## Environment Setup

### 1. Create a virtual environment (recommended)

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

### 3. Install dependencies

```bash
pip install torch torchvision numpy pillow
```

> **GPU (optional):** If you have an NVIDIA GPU and want CUDA support, install the matching PyTorch build from [pytorch.org](https://pytorch.org).

### 4. Verify

```bash
python -c "import torch; print(torch.__version__)"
```

## Quick Start

```bash
python generate_attack.py
```

Answer the prompts. It writes:

- `generated_attack.py` — the attack script
- `generated_config.json` — your choices

Then run:

```bash
python generated_attack.py
```

## Datasets

| Dataset | Description |
|---------|-------------|
| sklearn_digits | 8×8 grayscale digits (0-9). Uses local lab files. Fast. |
| MNIST | 28×28 grayscale handwritten digits. Downloads automatically. |
| CIFAR-10 | 32×32 color images, 10 classes. Downloads automatically. |
| CIFAR-100 | 32×32 color images, 100 classes. Downloads automatically. |
| ImageNet | Large-scale 1000-class images. Needs local data or uses a demo subset. |
| image_folder | Your own images in class subfolders (`0/`, `1/`, …). |
| custom | Edit `load_data()` yourself. |

## Models

| Model | Description |
|-------|-------------|
| mlp_digits | Simple MLP for 8×8 digits. |
| cnn_small | Small CNN with adaptive pooling (grayscale). |
| cnn_cifar | Small CNN for 32×32 color images (CIFAR). |
| resnet18 | torchvision ResNet-18 (ImageNet-style). |
| pretrained_custom | Load your own `.pt` weights (match the architecture). |
| custom | Paste your own model code. |

You can optionally provide a `.pt` weights file for any model.

## Single-image mode

When prompted, you can choose to attack **one uploaded image** instead of a full dataset.

- Provide the path to the image.
- Choose a target class.
- The generated script loads that image, shows the original prediction, runs the attack, and shows the new prediction.
