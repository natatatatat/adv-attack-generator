#!/usr/bin/env python3
"""
Adversarial Attack Script Generator

Interactive tool that generates a ready-to-run adversarial attack script.

Usage:
    python generate_attack.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ATTACKS = {
    "1": {"name": "FGSM", "description": "Fast Gradient Sign Method (single step)"},
    "2": {"name": "I-FGSM", "description": "Iterative FGSM"},
    "3": {"name": "PGD", "description": "Projected Gradient Descent"},
}

DATASETS = {
    "1": {
        "name": "sklearn_digits",
        "description": "8x8 grayscale digits (0-9). Uses local lab files in data/processed/. Small and fast.",
    },
    "2": {
        "name": "mnist",
        "description": "MNIST: 28x28 grayscale handwritten digits (0-9). 60k train / 10k test. Downloads automatically.",
    },
    "3": {
        "name": "cifar10",
        "description": "CIFAR-10: 32x32 color images, 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck). Downloads automatically.",
    },
    "4": {
        "name": "cifar100",
        "description": "CIFAR-100: 32x32 color images, 100 classes. Harder than CIFAR-10. Downloads automatically.",
    },
    "5": {
        "name": "imagenet",
        "description": "ImageNet: large-scale color images, 1000 classes. Requires a local folder or torchvision download (large). Use a small subset.",
    },
    "6": {
        "name": "image_folder",
        "description": "Your own folder of images organized in class subfolders (0/, 1/, 2/, ...).",
    },
    "7": {
        "name": "custom",
        "description": "You will edit the load_data() function yourself.",
    },
}

MODELS = {
    "1": {
        "name": "mlp_digits",
        "description": "Simple MLP for 8x8 digits (64 inputs → 10 classes). Good for sklearn digits.",
    },
    "2": {
        "name": "cnn_small",
        "description": "Small CNN with adaptive pooling. Works for grayscale (MNIST/digits) or can be adapted.",
    },
    "3": {
        "name": "cnn_cifar",
        "description": "Small CNN for 32x32 color images (CIFAR-10/100). 3 input channels.",
    },
    "4": {
        "name": "resnet18",
        "description": "ResNet-18 (torchvision). Common pretrained model for ImageNet or fine-tuned CIFAR.",
    },
    "5": {
        "name": "pretrained_custom",
        "description": "Load your own .pt weights. You must match the architecture in the generated script.",
    },
    "6": {
        "name": "custom",
        "description": "You will paste your own model architecture.",
    },
}


def prompt(msg: str, default: str | None = None) -> str:
    if default is not None:
        val = input(f"{msg} [{default}]: ").strip()
        return val if val else default
    return input(f"{msg}: ").strip()


def prompt_float(msg: str, default: float) -> float:
    while True:
        try:
            return float(prompt(msg, str(default)))
        except ValueError:
            print("Please enter a number.")


def prompt_int(msg: str, default: int) -> int:
    while True:
        try:
            return int(prompt(msg, str(default)))
        except ValueError:
            print("Please enter an integer.")


def choose(options: dict[str, dict], title: str) -> str:
    print(f"\n{title}")
    for key, info in options.items():
        print(f"  {key}. {info['name']} — {info['description']}")
    while True:
        choice = input("Select number: ").strip()
        if choice in options:
            return choice
        print("Invalid choice.")


def yes_no(msg: str, default: bool = False) -> bool:
    default_str = "Y/n" if default else "y/N"
    val = input(f"{msg} [{default_str}]: ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes")


def generate_script(config: dict[str, Any]) -> str:
    attack = config["attack"]
    targeted = config["targeted"]
    epsilon = config["epsilon"]
    steps = config.get("steps", 10)
    step_size = config.get("step_size", epsilon / 4)
    target_class = config.get("target_class")
    model_type = config["model_type"]
    dataset_type = config["dataset_type"]
    model_path = config.get("model_path", "")
    image_folder = config.get("image_folder", "")
    image_size = config.get("image_size", 28)
    single_image = config.get("single_image", False)
    single_image_path = config.get("single_image_path", "")
    single_image_size = config.get("single_image_size", 32)
    output_dir = config.get("output_dir", "artifacts/generated_attack")
    num_samples = config.get("num_samples", 200)

    # ---------- Model ----------
    if model_type == "mlp_digits":
        model_code = '''
def build_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
'''
    elif model_type == "cnn_small":
        model_code = '''
class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def build_model():
    return SmallCNN(in_channels=1, num_classes=10)
'''
    elif model_type == "cnn_cifar":
        model_code = '''
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def build_model():
    return SmallCNN(num_classes=10)
'''
    elif model_type == "resnet18":
        model_code = '''
def build_model():
    from torchvision import models
    # Downloads ImageNet pretrained weights if no local file is provided
    model = models.resnet18(weights="DEFAULT")
    return model
'''
    elif model_type == "pretrained_custom":
        model_code = '''
def build_model():
    # TODO: Define the architecture that matches your .pt weights.
    # Example:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
'''
    else:
        model_code = '''
def build_model():
    # TODO: Replace with your model architecture
    raise NotImplementedError("Provide your model")
'''

    # ---------- Data ----------
    if single_image:
        data_code = f'''
def load_data():
    from PIL import Image
    import torchvision.transforms as T
    img = Image.open("{single_image_path}").convert("RGB")
    transform = T.Compose([
        T.Resize(({single_image_size}, {single_image_size})),
        T.ToTensor(),
    ])
    x = transform(img).unsqueeze(0)          # shape (1, C, H, W)
    y = torch.tensor([0])                    # dummy label (not used for targeted)
    return x, y
'''
    elif dataset_type == "sklearn_digits":
        data_code = f'''
def load_data():
    """Load digits. Generates them with sklearn if lab files are missing."""
    from pathlib import Path
    x_path = Path("data/processed/x_test.npy")
    y_path = Path("data/processed/y_test.npy")
    if x_path.exists() and y_path.exists():
        x = torch.tensor(np.load(x_path)).float()[:{num_samples}]
        y = torch.tensor(np.load(y_path)).long()[:{num_samples}]
        return x, y
    print("Lab files not found. Generating digits with sklearn...")
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    x = digits.images.astype("float32") / 16.0
    x = x[:, None, :, :]
    y = digits.target.astype("int64")
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    n = min({num_samples}, len(x_test))
    return torch.tensor(x_test[:n]), torch.tensor(y_test[:n])
'''
    elif dataset_type == "mnist":
        data_code = f'''
def load_data():
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    n = min({num_samples}, len(test_set))
    x = torch.stack([test_set[i][0] for i in range(n)])
    y = torch.tensor([test_set[i][1] for i in range(n)])
    return x, y
'''
    elif dataset_type == "cifar10":
        data_code = f'''
def load_data():
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    n = min({num_samples}, len(test_set))
    x = torch.stack([test_set[i][0] for i in range(n)])
    y = torch.tensor([test_set[i][1] for i in range(n)])
    return x, y
'''
    elif dataset_type == "cifar100":
        data_code = f'''
def load_data():
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    n = min({num_samples}, len(test_set))
    x = torch.stack([test_set[i][0] for i in range(n)])
    y = torch.tensor([test_set[i][1] for i in range(n)])
    return x, y
'''
    elif dataset_type == "imagenet":
        data_code = f'''
def load_data():
    from torchvision import datasets, transforms
    # Uses a local ImageNet-style folder if present, otherwise a small subset via torchvision
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    try:
        test_set = datasets.ImageNet(root="./data", split="val", transform=transform)
    except Exception:
        print("ImageNet not found locally. Using a tiny random subset for demo.")
        x = torch.rand({num_samples}, 3, 224, 224)
        y = torch.randint(0, 1000, ({num_samples},))
        return x, y
    n = min({num_samples}, len(test_set))
    x = torch.stack([test_set[i][0] for i in range(n)])
    y = torch.tensor([test_set[i][1] for i in range(n)])
    return x, y
'''
    elif dataset_type == "image_folder":
        data_code = f'''
def load_data():
    from pathlib import Path
    from PIL import Image
    import torchvision.transforms as T

    folder = Path("{image_folder}")
    size = {image_size}
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])

    images, labels = [], []
    class_dirs = [d for d in folder.iterdir() if d.is_dir() and d.name.isdigit()]
    if not class_dirs:
        raise FileNotFoundError(f"No class subfolders (0/, 1/, ...) found in {{folder}}")

    for class_dir in sorted(class_dirs, key=lambda p: int(p.name)):
        label = int(class_dir.name)
        for img_path in list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")):
            img = Image.open(img_path).convert("RGB")
            images.append(transform(img))
            labels.append(label)
            if len(images) >= {num_samples}:
                break
        if len(images) >= {num_samples}:
            break

    if not images:
        raise FileNotFoundError(f"No images found in {{folder}}")

    return torch.stack(images), torch.tensor(labels)
'''
    else:
        data_code = '''
def load_data():
    # TODO: Return (x, y) tensors. x should be in [0, 1].
    raise NotImplementedError("Provide your data loading code")
'''

    # ---------- Attack ----------
    target_tensor = f"torch.full_like(y, {target_class})" if targeted else "y"
    sign = "-" if targeted else "+"

    if attack == "FGSM":
        attack_fn = f'''
def attack(model, x, y, epsilon={epsilon}):
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, {target_tensor})
    model.zero_grad()
    loss.backward()
    grad_sign = x.grad.sign()
    x_adv = x {sign} epsilon * grad_sign
    return torch.clamp(x_adv, 0, 1).detach()
'''
    else:  # I-FGSM and PGD
        attack_fn = f'''
def attack(model, x, y, epsilon={epsilon}, step_size={step_size}, steps={steps}):
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = nn.CrossEntropyLoss()(logits, {target_tensor})
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv {sign} step_size * grad_sign
            delta = torch.clamp(x_adv - x, -epsilon, epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)
        x_adv = x_adv.detach()
    return x_adv
'''

    load_weights = ""
    if model_path:
        load_weights = f'''
    if Path("{model_path}").exists():
        model.load_state_dict(torch.load("{model_path}", map_location="cpu"))
        print("Loaded weights from {model_path}")
    else:
        print("Weights file not found. Training a quick model instead...")
        train_quick(model, x, y)
'''
    else:
        # No weights provided: train quickly unless it's already a pretrained torchvision model
        if model_type != "resnet18":
            load_weights = '''
    print("No weights file provided. Training a quick model on the data...")
    train_quick(model, x, y)
'''

    train_fn = '''
def train_quick(model, x, y, epochs=3, lr=0.01):
    """Short training so the model is not random."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    # Use a small subset if data is large
    n = min(len(x), 500)
    x_t, y_t = x[:n], y[:n]
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        opt.step()
        print(f"  quick train epoch {epoch+1}/{epochs} loss={loss.item():.4f}")
    model.eval()
'''

    script = f'''#!/usr/bin/env python3
"""
Generated Adversarial Attack Script
Attack : {attack}
Targeted: {targeted}
Epsilon: {epsilon}
{"Target class: " + str(target_class) if targeted else ""}
Dataset: {dataset_type}
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json

OUTPUT_DIR = Path("{output_dir}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

{model_code}

{data_code}

{attack_fn}

{train_fn}

def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(1)
        return (preds == y).float().mean().item()

def main():
    print("Loading data...")
    x, y = load_data()
    print(f"Loaded {{len(x)}} sample(s), shape {{tuple(x.shape)}}")

    model = build_model()
{load_weights}
    model.eval()

    # Original prediction
    with torch.no_grad():
        orig_pred = model(x).argmax(1)
    print(f"Original prediction(s): {{orig_pred.tolist()}}")

    clean_acc = evaluate(model, x, y)
    print(f"Clean accuracy (vs labels): {{clean_acc:.4f}}")

    print("Running attack...")
    x_adv = attack(model, x, y)

    with torch.no_grad():
        adv_pred = model(x_adv).argmax(1)
    print(f"Adversarial prediction(s): {{adv_pred.tolist()}}")

    adv_acc = evaluate(model, x_adv, y)
    print(f"Adversarial accuracy: {{adv_acc:.4f}}")
    print(f"Drop: {{clean_acc - adv_acc:.4f}}")

    np.save(OUTPUT_DIR / "x_adv.npy", x_adv.cpu().numpy())
    np.save(OUTPUT_DIR / "y.npy", y.cpu().numpy())
    summary = {{
        "attack": "{attack}",
        "targeted": {str(targeted)},
        "epsilon": {epsilon},
        "original_predictions": orig_pred.tolist(),
        "adversarial_predictions": adv_pred.tolist(),
        "clean_accuracy": clean_acc,
        "adversarial_accuracy": adv_acc,
        "num_samples": len(x),
    }}
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved results to {{OUTPUT_DIR}}")

if __name__ == "__main__":
    main()
'''
    return script


def main():
    print("=" * 60)
    print(" Adversarial Attack Script Generator")
    print("=" * 60)

    attack_choice = choose(ATTACKS, "Choose attack type:")
    attack = ATTACKS[attack_choice]["name"]

    targeted = yes_no("Targeted attack?", default=False)
    target_class = None
    if targeted:
        target_class = prompt_int("Target class", 7)

    epsilon = prompt_float("Max perturbation epsilon (0.0-1.0)", 0.1)

    steps = 10
    step_size = epsilon / 4
    if attack in ("I-FGSM", "PGD"):
        steps = prompt_int("Number of steps", 10)
        step_size = prompt_float("Step size", round(epsilon / 4, 4))

    model_choice = choose(MODELS, "Choose model:")
    model_type = MODELS[model_choice]["name"]
    model_path = ""

    # Optional weights for any model
    if model_type in ("pretrained_custom", "resnet18") or yes_no("Load model weights from a .pt file?", default=False):
        model_path = prompt("Path to weights file (.pt)", "")

    dataset_choice = choose(DATASETS, "Choose dataset:")
    dataset_type = DATASETS[dataset_choice]["name"]

    # Single-image mode
    single_image = yes_no("Attack a single uploaded image instead of a dataset?", default=False)
    single_image_path = ""
    single_image_size = 32
    if single_image:
        single_image_path = prompt("Path to the image file")
        single_image_size = prompt_int("Resize image to this size (square)", 32)
        if not targeted:
            print("Single-image mode works best as a targeted attack.")
            if yes_no("Switch to targeted?", default=True):
                targeted = True
                target_class = prompt_int("Target class", 0)

    image_folder = ""
    image_size = 28
    if dataset_type == "image_folder" and not single_image:
        image_folder = prompt("Path to image folder (with class subfolders 0/, 1/, ...)")
        image_size = prompt_int("Resize images to this size", 28)

    num_samples = 1 if single_image else prompt_int("Max number of samples to attack", 200)
    output_dir = prompt("Output directory for results", "artifacts/generated_attack")

    config = {
        "attack": attack,
        "targeted": targeted,
        "target_class": target_class,
        "epsilon": epsilon,
        "steps": steps,
        "step_size": step_size,
        "model_type": model_type,
        "model_path": model_path,
        "dataset_type": dataset_type,
        "image_folder": image_folder,
        "image_size": image_size,
        "single_image": single_image,
        "single_image_path": single_image_path,
        "single_image_size": single_image_size,
        "num_samples": num_samples,
        "output_dir": output_dir,
    }

    script = generate_script(config)
    out_path = Path("generated_attack.py")
    out_path.write_text(script)
    print(f"\nGenerated script written to: {out_path.resolve()}")
    print("Run it with: python generated_attack.py")

    Path("generated_config.json").write_text(json.dumps(config, indent=2))
    print("Config saved to generated_config.json")


if __name__ == "__main__":
    main()
