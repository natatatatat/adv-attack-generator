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

## Example: CIFAR-10 that actually fools the model

This example uses PGD on CIFAR-10. With these settings the attack changes most predictions.

### Every prompt (copy these answers)

```
Choose attack type:                          3
Targeted attack? [y/N]:                      n
Max perturbation epsilon (0.0-1.0) [0.1]:    0.1
Number of steps [10]:                        20
Step size [0.025]:                           0.01
Choose model:                                3
Load weights from a .pt file? [y/N]:         n
Choose dataset:                              3
Attack a single uploaded image? [y/N]:       n
Max number of samples to attack [200]:       50
```

### Run it

```bash
python generate_attack.py
# enter the answers above

python generated_attack.py
```

CIFAR-10 downloads automatically.

### Why the first try may not fool the model

The generated script only quick-trains for 8 epochs. That is often too weak on CIFAR-10.

**After generating the script, open `generated_attack.py` and change:**

```python
def train_quick(model, x, y, epochs=8, lr=0.001):
```

to:

```python
def train_quick(model, x, y, epochs=20, lr=0.001):
```

Then run `python generated_attack.py` again.

### What you should see

- Clean accuracy around 30–50%
- Adversarial accuracy much lower (often near 0%)
- Line like: `Result: attack changed the prediction on 50%+ of samples.`
- `Attack was effective.`

### Results folder

```
orig_samples/            # original images
adversarial_samples/     # adversarial images
before_after.png         # top = original, bottom = adversarial
summary.json
```

### CIFAR-10 class numbers (if you use a targeted attack)

| Number | Class     | Number | Class  |
|--------|-----------|--------|--------|
| 0      | airplane  | 5      | dog    |
| 1      | automobile| 6      | frog   |
| 2      | bird      | 7      | horse  |
| 3      | cat       | 8      | ship   |
| 4      | deer      | 9      | truck  |

For a targeted example, answer `y` to “Targeted attack?” and enter a class number (e.g. `5` for dog).
