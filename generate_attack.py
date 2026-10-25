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
    "2": {"name": "I-FGSM/BIM", "description": "Iterative FGSM / Basic Iterative Method"},
    "3": {"name": "PGD", "description": "Projected Gradient Descent (BIM with random start)"},
    "4": {"name": "DeepFool", "description": "Minimal perturbation to cross the decision boundary"},
}

# Compatible pairs (model_type, dataset_type)
COMPATIBLE = {
    "mlp_digits": {"sklearn_digits"},
    "cnn_small": {"mnist", "sklearn_digits"},
    "cnn_cifar": {"cifar10", "cifar100"},
    "resnet18": {"imagenet", "cifar10", "cifar100", "image_folder"},
    "pretrained_custom": {"sklearn_digits", "mnist", "cifar10", "cifar100", "imagenet", "image_folder"},
    "custom": {"sklearn_digits", "mnist", "cifar10", "cifar100", "imagenet", "image_folder"},
}

def is_compatible(model_type: str, dataset_type: str) -> bool:
    return dataset_type in COMPATIBLE.get(model_type, set())

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
        "name": "h5_keras",
        "description": "Load a Keras/TensorFlow .h5 model file (requires tensorflow).",
    },
    "6": {
        "name": "pretrained_custom",
        "description": "Load your own .pt weights. You must match the architecture in the generated script.",
    },
    "7": {
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
    output_dir = config.get("output_dir", ".")
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
        # Grayscale datasets need 1 channel; color datasets need 3
        if dataset_type in ("mnist", "sklearn_digits") or config.get("color_mode") == "L":
            in_ch = 1
        else:
            in_ch = 3
        model_code = f'''
class SmallCNN(nn.Module):
    def __init__(self, in_channels={in_ch}, num_classes=10):
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
    return SmallCNN(in_channels={in_ch}, num_classes=10)
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
    from torchvision import models, transforms
    model = models.resnet18(weights="DEFAULT")
    # ImageNet normalization so pretrained weights work correctly
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    class NormalizedResNet(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
        def forward(self, x):
            x = (x - self.mean) / self.std
            return self.base(x)

    return NormalizedResNet(model)
'''
    elif model_type == "h5_keras":
        model_code = '''
def build_model():
    # Keras .h5 models are loaded separately
    return None
'''
    elif model_type == "pretrained_custom":
        model_code = '''
def build_model():
    # USER: Update this section to match your .pt model architecture.
    # Example (edit input size and layers to match your model):
    # return nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(64, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 10),
    # )
    raise NotImplementedError(
        "Update build_model() with your architecture before running. "
        "See the example in the comments above."
    )
'''
    else:
        model_code = '''
def build_model():
    # USER: Update this section with your model code.
    # Example:
    # class MyModel(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    #     def forward(self, x):
    #         return self.net(x)
    # return MyModel()
    raise NotImplementedError(
        "Update build_model() with your model before running. "
        "See the example in the comments above."
    )
'''

    # ---------- Data ----------
    if single_image:
        color_mode = config.get("color_mode", "RGB")
        data_code = f'''
def load_data():
    from PIL import Image
    import torchvision.transforms as T
    img = Image.open("{single_image_path}")
    mode = "{color_mode}"
    size = {single_image_size}
    if mode == "L":
        img = img.convert("L")
    elif mode == "RGB":
        img = img.convert("RGB")
    # else keep original mode
    transforms = []
    if size > 0:
        transforms.append(T.Resize((size, size)))
    transforms.append(T.ToTensor())
    transform = T.Compose(transforms)
    x = transform(img).unsqueeze(0)
    y = torch.tensor([0])
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
    elif attack == "DeepFool":
        attack_fn = '''
def attack(model, x, y, max_iter=20, overshoot=0.02):
    """DeepFool: small perturbation that changes the prediction."""
    model.eval()
    x_adv = x.clone().detach()
    for i in range(x_adv.shape[0]):
        xi = x_adv[i:i+1].clone().detach().requires_grad_(True)
        for _ in range(max_iter):
            logits = model(xi)
            pred = int(logits.argmax(1).item())
            model.zero_grad()
            logits[0, pred].backward(retain_graph=True)
            grad_pred = xi.grad.detach().clone()
            best_pert = float("inf")
            r_best = None
            for k in range(logits.shape[1]):
                if k == pred:
                    continue
                if xi.grad is not None:
                    xi.grad.zero_()
                logits[0, k].backward(retain_graph=True)
                w = xi.grad.detach().clone() - grad_pred
                f = float((logits[0, k] - logits[0, pred]).item())
                pert = abs(f) / (w.norm() + 1e-8)
                if pert < best_pert:
                    best_pert = pert
                    r_best = (abs(f) / (w.norm() ** 2 + 1e-8)) * w
            if r_best is None:
                break
            xi = torch.clamp(xi + (1 + overshoot) * r_best, 0, 1).detach().requires_grad_(True)
            if int(model(xi).argmax(1).item()) != pred:
                break
        x_adv[i] = xi.detach().clamp(0, 1)
    return x_adv
'''
    else:  # I-FGSM/BIM and PGD
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
    if model_type == "h5_keras" and model_path:
        load_weights = f'''
    import tensorflow as tf
    print("Loading Keras .h5 model from {model_path}...")
    model = tf.keras.models.load_model("{model_path}")
    print("Loaded .h5 model. Attack will use TensorFlow gradients.")
'''
    elif model_path:
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
def train_quick(model, x, y, epochs=20, lr=0.001):
    """Train longer so the model learns real patterns."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = min(len(x), 1000)
    x_t, y_t = x[:n], y[:n]
    print(f"Quick-training on {n} samples for {epochs} epochs...")
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        opt.step()
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs} loss={loss.item():.4f}")
    model.eval()
    print("Quick training finished.")
'''

    script = f'''#!/usr/bin/env python3
"""
Generated Adversarial Attack Script
Attack : {attack}
Targeted: {targeted}
Epsilon: {epsilon}
{"Target class number: " + str(target_class) if targeted else ""}

CLASS NUMBERS:
Update the target class number to match your model.
Examples:
  CIFAR-10: 0=airplane, 1=car, 3=cat, 5=dog, 7=horse
  ImageNet: look up the class index for the label you want
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
    drop = clean_acc - adv_acc
    print(f"Accuracy drop: {{drop:.4f}}")

    # Plain-language result
    if len(x) == 1:
        print(f"\\nResult: image changed from class {{orig_pred.item()}} to class {{adv_pred.item()}}")
        if orig_pred.item() != adv_pred.item():
            print("Attack succeeded on this image.")
        else:
            print("Attack did not change the prediction. Try a larger epsilon.")
    else:
        success_rate = (orig_pred != adv_pred).float().mean().item()
        print(f"\\nResult: attack changed the prediction on {{success_rate:.0%}} of samples.")
        if drop > 0.1:
            print("Attack was effective.")
        else:
            print("Attack had limited effect. Try a larger epsilon or more steps.")

    # Save individual images
    orig_dir = OUTPUT_DIR / "orig_samples"
    adv_dir = OUTPUT_DIR / "adversarial_samples"
    orig_dir.mkdir(parents=True, exist_ok=True)
    adv_dir.mkdir(parents=True, exist_ok=True)

    def save_images(tensor, folder, prefix):
        from PIL import Image
        for i in range(len(tensor)):
            img = tensor[i].cpu()
            if img.shape[0] == 1:
                img = img.squeeze(0)
                arr = (img.numpy() * 255).clip(0, 255).astype("uint8")
                pil = Image.fromarray(arr, mode="L")
            else:
                arr = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
                pil = Image.fromarray(arr, mode="RGB")
            # Scale up small images (e.g. 32x32 CIFAR) so they are easy to view
            if pil.width < 128:
                scale = max(1, 128 // pil.width)
                pil = pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)
            pil.save(folder / f"{{prefix}}_{{i:04d}}.png")

    save_images(x, orig_dir, "orig")
    save_images(x_adv, adv_dir, "adv")
    print(f"Saved original images to {{orig_dir}}")
    print(f"Saved adversarial images to {{adv_dir}}")

    # Side-by-side comparison (scaled up so small images are easy to see)
    try:
        from PIL import Image as PILImage
        n_show = min(5, len(x))
        scale = 8  # make 32x32 look like 256x256
        thumbs = []
        for i in range(n_show):
            def to_pil(t):
                im = t.cpu()
                if im.shape[0] == 1:
                    arr = (im.squeeze(0).numpy() * 255).clip(0, 255).astype("uint8")
                    pil = PILImage.fromarray(arr, mode="L").convert("RGB")
                else:
                    arr = (im.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
                    pil = PILImage.fromarray(arr, mode="RGB")
                return pil.resize((pil.width * scale, pil.height * scale), PILImage.NEAREST)
            thumbs.append((to_pil(x[i]), to_pil(x_adv[i])))
        w, h = thumbs[0][0].size
        grid = PILImage.new("RGB", (w * n_show, h * 2), (255, 255, 255))
        for i, (orig, adv) in enumerate(thumbs):
            grid.paste(orig, (i * w, 0))
            grid.paste(adv, (i * w, h))
        out_path = Path("before_after.png").resolve()
        grid.save(out_path)
        print(f"Saved comparison grid to {{out_path}}")
        print("Top row = original, bottom row = adversarial")
    except Exception as e:
        print(f"Could not save comparison grid: {{e}}")

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
    print(f"\\nSaved results to {{OUTPUT_DIR}}")
    print("Next: open the folder above to see adversarial examples and summary.json")

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
        print("Class numbers depend on the model.")
        print("Example: for CIFAR-10, 0=plane, 1=car, 3=cat, 5=dog, 7=horse.")
        print("Example: for ImageNet, look up the class index (e.g. 207 can be a dog breed).")
        print("Update the target class number below to match your model.")
        target_class = prompt_int("Target class number", 7)

    epsilon = prompt_float("Max perturbation epsilon (0.0-1.0)", 0.1)

    steps = 10
    step_size = epsilon / 4
    if attack in ("I-FGSM/BIM", "PGD"):
        steps = prompt_int("Number of steps", 10)
        step_size = prompt_float("Step size", round(epsilon / 4, 4))

    # Model selection
    print("\nModel options:")
    print("  - Pick a built-in model (MLP, CNN, ResNet-18)")
    print("  - ResNet-18 will download pretrained ImageNet weights if you don't provide a file")
    print("  - Or provide your own .pt / .h5 file")
    model_choice = choose(MODELS, "Choose model:")
    model_type = MODELS[model_choice]["name"]
    model_path = ""

    if model_type == "h5_keras":
        model_path = prompt("Path to .h5 model file")
        print("Note: .h5 models are loaded with TensorFlow. The attack will use a TensorFlow gradient path.")
    elif model_type == "resnet18":
        if yes_no("Download pretrained ImageNet weights for ResNet-18?", default=True):
            model_path = ""  # signals download
        else:
            model_path = prompt("Path to .pt weights (or leave blank)", "")
    elif model_type in ("pretrained_custom",) or yes_no("Load weights from a .pt file?", default=False):
        model_path = prompt("Path to .pt weights file", "")

    dataset_choice = choose(DATASETS, "Choose dataset:")
    dataset_type = DATASETS[dataset_choice]["name"]

    # Warn or block incompatible combinations
    if not is_compatible(model_type, dataset_type):
        print(f"\nWarning: '{model_type}' is not designed for '{dataset_type}'.")
        print("This will likely crash. Recommended pairs:")
        print("  mlp_digits  → sklearn_digits")
        print("  cnn_small   → mnist")
        print("  cnn_cifar   → cifar10 / cifar100")
        print("  resnet18    → imagenet or cifar")
        if not yes_no("Continue anyway?", default=False):
            print("Aborting. Please re-run and choose a matching pair.")
            return

    # Single-image mode
    single_image = yes_no("Attack a single uploaded image instead of a dataset?", default=False)
    single_image_path = ""
    single_image_size = 32
    if single_image:
        if model_type == "mlp_digits":
            print("Note: mlp_digits only works with 8x8 images.")
            if yes_no("Use ResNet-18 (downloads pretrained weights) instead?", default=True):
                model_type = "resnet18"
                model_path = ""
            elif yes_no("Use a flexible CNN instead?", default=True):
                model_type = "cnn_small"
        if model_type not in ("resnet18", "h5_keras", "pretrained_custom") and not model_path:
            print("Warning: no pretrained model selected. Single-image attacks work best with a pretrained model.")
            if yes_no("Download ResNet-18 pretrained weights?", default=True):
                model_type = "resnet18"
                model_path = ""
        single_image_path = prompt("Path to the image file")
        if yes_no("Resize the image?", default=True):
            single_image_size = prompt_int("Resize to this square size", 32)
        else:
            single_image_size = 0  # 0 means keep original size
        color_choice = choose(
            {"1": {"name": "RGB", "description": "Color (3 channels)"},
             "2": {"name": "L", "description": "Grayscale (1 channel)"},
             "3": {"name": "keep", "description": "Keep original color mode"}},
            "Image color mode:"
        )
        color_mode = {"1": "RGB", "2": "L", "3": "keep"}[color_choice]
        if not targeted:
            print("You can force a specific target class, or just make the model predict something else.")
            if yes_no("Force a specific target class?", default=False):
                targeted = True
                target_class = prompt_int("Target class number", 0)
            else:
                print("Will try to change the prediction away from the original class.")
    else:
        color_mode = "RGB"
        single_image_size = 32

    image_folder = ""
    image_size = 28
    if dataset_type == "image_folder" and not single_image:
        image_folder = prompt("Path to image folder (with class subfolders 0/, 1/, ...)")
        image_size = prompt_int("Resize images to this size", 28)

    num_samples = 1 if single_image else prompt_int("Max number of samples to attack", 200)
    output_dir = "."

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
        "color_mode": color_mode,
        "num_samples": num_samples,
        "output_dir": output_dir,
    }

    script = generate_script(config)
    out_path = Path("generated_attack.py")
    out_path.write_text(script)
    print(f"\nGenerated script written to: {out_path.resolve()}")
    print("Next step: run it with")
    print(f"  python {out_path.name}")

    # Create result folders now so they exist before the attack runs
    out = Path(output_dir).resolve()
    (out / "orig_samples").mkdir(parents=True, exist_ok=True)
    (out / "adversarial_samples").mkdir(parents=True, exist_ok=True)
    print(f"\nCreated folders:")
    print(f"  {out / 'orig_samples'}")
    print(f"  {out / 'adversarial_samples'}")
    print("Run the attack script to fill them with images.")
    print("A before_after.png comparison will also be saved in:")
    print(f"  {out}")

    Path("generated_config.json").write_text(json.dumps(config, indent=2))
    print("Config saved to generated_config.json")


if __name__ == "__main__":
    main()
