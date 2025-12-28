#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Generate visual reports for SpaceNet.")
    parser.add_argument("--data-root", default="./img")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir", default="./reports")
    parser.add_argument("--max-grid", type=int, default=16)
    parser.add_argument("--csv-log", default="./logs/train_metrics.csv")
    return parser.parse_args()


def build_model(name, num_classes):
    if not hasattr(models, name):
        raise ValueError(f"Unknown torchvision model: {name}")
    model_fn = getattr(models, name)
    model = model_fn(weights=None)
    if hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Sequential):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    elif hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Unsupported model head; add a head mapping here.")
    return model


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)


def filter_bad_samples(samples):
    good = []
    for path, target in samples:
        try:
            with Image.open(path) as img:
                img.verify()
            good.append((path, target))
        except (UnidentifiedImageError, OSError):
            continue
    return good


def build_val_split(dataset, val_split, seed, transform):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    split_gen = torch.Generator().manual_seed(seed)
    _, val_idx = random_split(list(range(len(dataset))), [train_size, val_size], generator=split_gen)
    val_ds = datasets.ImageFolder(dataset.root, transform=transform)
    val_ds.samples = dataset.samples
    val_ds.imgs = dataset.samples
    return Subset(val_ds, val_idx.indices)


def find_latest_checkpoint(path_prefix):
    candidates = sorted(Path(path_prefix).glob("spacenet_*"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def compute_confusion(model, loader, num_classes, device):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            for t, p in zip(targets, preds):
                cm[t.long(), p.long()] += 1
    return cm


def save_confusion_matrix(cm, classes, path):
    import matplotlib.pyplot as plt

    cm_np = cm.float()
    cm_norm = cm_np / cm_np.sum(dim=1, keepdim=True).clamp_min(1.0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm.numpy(), cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_class_metrics(cm, classes, path):
    precision = []
    recall = []
    f1 = []
    for i, cls in enumerate(classes):
        tp = cm[i, i].item()
        fp = cm[:, i].sum().item() - tp
        fn = cm[i, :].sum().item() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1_score = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1"])
        for cls, p, r, f1s in zip(classes, precision, recall, f1):
            writer.writerow([cls, f"{p:.4f}", f"{r:.4f}", f"{f1s:.4f}"])


def save_sample_grid(model, dataset, classes, path, max_images, device):
    import matplotlib.pyplot as plt

    model.eval()
    n = min(max_images, len(dataset))
    cols = int(n**0.5)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    idx = 0
    with torch.no_grad():
        for r in range(rows):
            for c in range(cols):
                ax = axes[r][c]
                ax.axis("off")
                if idx >= n:
                    continue
                image, label = dataset[idx]
                logits = model(image.unsqueeze(0).to(device))
                pred = logits.argmax(dim=1).item()
                img = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.set_title(f"t:{classes[label]} p:{classes[pred]}", fontsize=8)
                idx += 1

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_curves(csv_path, out_path):
    import matplotlib.pyplot as plt

    if not Path(csv_path).exists():
        return False
    epochs = []
    train_acc = []
    val_acc = []
    train_loss = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, train_loss, label="train_loss", color="tab:red")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_acc, label="train_acc", color="tab:blue")
    ax2.plot(epochs, val_acc, label="val_acc", color="tab:green")
    ax2.set_ylabel("accuracy", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = args.checkpoint
    if not ckpt_path:
        latest = find_latest_checkpoint("./checkpoints")
        if not latest:
            raise FileNotFoundError("No spacenet_* checkpoints found in ./checkpoints")
        ckpt_path = str(latest)

    val_tfms = transforms.Compose(
        [
            transforms.Resize(args.image_size + 32),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    full_ds = datasets.ImageFolder(args.data_root, transform=val_tfms)
    full_ds.samples = filter_bad_samples(full_ds.samples)
    full_ds.imgs = full_ds.samples
    val_ds = build_val_split(full_ds, args.val_split, args.seed, val_tfms)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes=len(full_ds.classes)).to(device)
    load_checkpoint(model, ckpt_path)
    model.eval()

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cm = compute_confusion(model, val_loader, len(full_ds.classes), device)
    save_confusion_matrix(cm, full_ds.classes, out_dir / "confusion_matrix.png")
    save_class_metrics(cm, full_ds.classes, out_dir / "class_metrics.csv")
    save_sample_grid(model, val_ds, full_ds.classes, out_dir / "sample_grid.png", args.max_grid, device)
    save_curves(args.csv_log, out_dir / "training_curves.png")

    print(f"Saved reports to {out_dir}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
