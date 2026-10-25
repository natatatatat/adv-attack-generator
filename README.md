# Adversarial Attack Script Generator

Interactive tool that generates a ready-to-run adversarial attack script.

## Environment Setup

### Standard setup (most systems)

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

```bash
pip install --upgrade pip
pip install torch torchvision numpy pillow scikit-learn matplotlib
```

### Kali Linux on Apple Silicon (M1/M2/M3/M5) — important

If you see:

```
zsh: illegal hardware instruction  python -c "import torch; ..."
```

Use **Python 3.13** (3.14 wheels are unstable on aarch64) and set the OpenBLAS workaround:

```bash
# Install Python 3.13
sudo apt update
sudo apt install python3.13 python3.13-venv python3.13-dev -y

# Create environment
python3.13 -m venv venv
source venv/bin/activate

# Install packages
pip install --upgrade pip
pip install torch torchvision numpy pillow scikit-learn matplotlib

# Make the OpenBLAS fix permanent
echo 'export OPENBLAS_CORETYPE=ARMV8' >> venv/bin/activate
deactivate
source venv/bin/activate
```

### Verify

```bash
python -c "import torch; print(torch.__version__); print('OK')"
```

> **GPU note:** On Apple Silicon, GPU (MPS) only works on macOS, not inside Kali. PyTorch in Kali uses the CPU.

## Quick Start

```bash
python generate_attack.py
```

Answer the prompts. It writes:

- `generated_attack.py` — the attack script
- `generated_config.json` — your choices

It also creates these folders in the **same directory**:

- `orig_samples/` — will hold the original images
- `adversarial_samples/` — will hold the adversarial images

Then run:

```bash
python generated_attack.py
```

After the attack finishes you will have:

```
.
├── generate_attack.py
├── generated_attack.py
├── orig_samples/            # original images
├── adversarial_samples/     # adversarial images
├── before_after.png         # side-by-side comparison
└── summary.json             # accuracy before and after
```

## Attacks

| Attack | Description |
|--------|-------------|
| FGSM | Single-step Fast Gradient Sign Method |
| I-FGSM / BIM | Iterative FGSM (also called Basic Iterative Method) |
| PGD | Projected Gradient Descent (BIM with random start) |
| DeepFool | Finds a small perturbation that changes the prediction |

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

- Provide an image path.
- Resizing is optional (choose a size or keep the original).
- Color mode: RGB, grayscale, or keep original.
- Optionally force a specific target class, or just change the prediction to anything else.
