# Adversarial Attack Script Generator

Interactive tool that generates a ready-to-run adversarial attack script from simple prompts.  
No need to write the attack code yourself — choose the attack, model, and dataset, then run the generated script.

> **Note:** This is a personal project for learning and authorized testing only.  
> It is not related to my employer and was built outside of work.  
> Use it only on models and data you are allowed to test.

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
pip install torch torchvision numpy pillow scikit-learn
```

Pillow is enough for the before/after images. Matplotlib is optional.

### Kali Linux on Apple Silicon (M1/M2/M3/M5)

If you see:

```
zsh: illegal hardware instruction  python -c "import torch; ..."
```

Use **Python 3.13** (3.14 wheels are unstable on aarch64) and set the OpenBLAS workaround:

```bash
sudo apt update
sudo apt install python3.13 python3.13-venv python3.13-dev -y

python3.13 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch torchvision numpy pillow scikit-learn

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

- `orig_samples/` — original images
- `adversarial_samples/` — adversarial images

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
├── before_after.png         # top = original, bottom = adversarial
└── summary.json             # accuracy before and after
```

`before_after.png` is saved next to the script (Pillow only, no matplotlib required).

## Attacks

| Attack | Description |
|--------|-------------|
| FGSM | Single-step Fast Gradient Sign Method |
| I-FGSM / BIM | Iterative FGSM (also called Basic Iterative Method) |
| PGD | Projected Gradient Descent (BIM with random start) |
| DeepFool | Finds a small perturbation that changes the prediction |

## Datasets

| Dataset | Description |
|---------|-------------|
| sklearn_digits | 8×8 grayscale digits (0–9). Uses local lab files if present, otherwise generates them. |
| MNIST | 28×28 grayscale handwritten digits. Downloads automatically. |
| CIFAR-10 | 32×32 color images, 10 classes. Downloads automatically. |
| CIFAR-100 | 32×32 color images, 100 classes. Downloads automatically. |
| ImageNet | Large-scale 1000-class images. Needs local data or uses a small demo subset. |
| image_folder | Your own images in class subfolders (`0/`, `1/`, …). |
| custom | Edit `load_data()` yourself. |

## Models

| Model | Description |
|-------|-------------|
| mlp_digits | Simple MLP for 8×8 digits. |
| cnn_small | Small CNN with adaptive pooling (grayscale). |
| cnn_cifar | Small CNN for 32×32 color images (CIFAR). |
| resnet18 | torchvision ResNet-18. Downloads ImageNet weights if you do not provide a file. |
| h5_keras | Load a Keras/TensorFlow `.h5` model (requires TensorFlow). |
| pretrained_custom | Load your own `.pt` weights (you must match the architecture). |
| custom | Paste your own model code. |

You can provide a `.pt` weights file for most models. ResNet-18 can download pretrained weights for you.

## Single-image mode

- Provide a path to one image.
- Resizing is optional (choose a size or keep the original).
- Color mode: RGB, grayscale, or keep original.
- Optionally force a specific target class, or just change the prediction to anything else.

## Example: CIFAR-10 that fools the model

This example uses PGD on CIFAR-10. With these settings the attack changes most predictions.

### Every prompt

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

CIFAR-10 downloads automatically into a `data/` folder the first time.

### If the attack looks weak

The script quick-trains the model. If predictions barely change, open `generated_attack.py` and increase training:

```python
def train_quick(model, x, y, epochs=20, lr=0.001):
```

Change `20` to `30` if needed, then run the script again.

### What you should see

- Clean accuracy around 30–50%
- Adversarial accuracy much lower (often near 0%)
- A line like: `Result: attack changed the prediction on 50%+ of samples.`
- `Attack was effective.`

### Results

```
orig_samples/            # original images
adversarial_samples/     # adversarial images
before_after.png         # top row = original, bottom row = adversarial
summary.json
```

### CIFAR-10 class numbers (for targeted attacks)

| Number | Class      | Number | Class  |
|--------|------------|--------|--------|
| 0      | airplane   | 5      | dog    |
| 1      | automobile | 6      | frog   |
| 2      | bird       | 7      | horse  |
| 3      | cat        | 8      | ship   |
| 4      | deer       | 9      | truck  |

For a targeted attack, answer `y` to “Targeted attack?” and enter a class number (for example `5` for dog).
