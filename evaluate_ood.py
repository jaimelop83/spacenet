#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="OOD evaluation for SpaceNet classifiers.")
    parser.add_argument("--id-root", required=True, help="In-distribution dataset root.")
    parser.add_argument("--ood-root", required=True, help="Out-of-distribution dataset root.")
    parser.add_argument("--checkpoint", required=True, help="Path to classifier checkpoint.")
    parser.add_argument("--model", default="convnext_tiny", help="torchvision model name.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--metric",
        choices=["max_softmax", "energy"],
        default="max_softmax",
        help="Score used for OOD detection.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Energy temperature.")
    parser.add_argument("--ood-flat", action="store_true", help="Treat ood-root as a flat folder.")
    parser.add_argument("--plot", default=None, help="Path to save ROC curve plot (png).")
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


class FlatImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = self._collect_images(self.root)

    @staticmethod
    def _collect_images(root):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        files = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                files.append(path)
        return sorted(files)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)


def compute_scores(model, loader, metric, temperature, device):
    scores = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            if metric == "max_softmax":
                probs = torch.softmax(logits, dim=1)
                score = probs.max(dim=1).values
            else:
                score = -torch.logsumexp(logits / temperature, dim=1)
            scores.append(score.detach().cpu())
    return torch.cat(scores, dim=0)


def roc_auc(scores, labels):
    # labels: 1 for ID, 0 for OOD
    sorted_idx = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_idx]
    tps = torch.cumsum(sorted_labels, dim=0)
    fps = torch.cumsum(1 - sorted_labels, dim=0)
    tpr = tps / tps[-1].clamp_min(1)
    fpr = fps / fps[-1].clamp_min(1)
    auc = torch.trapz(tpr, fpr).item()
    return auc, tpr, fpr


def save_roc_plot(fpr, tpr, path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for --plot") from exc
    plt.figure(figsize=(5, 5))
    plt.plot(fpr.numpy(), tpr.numpy(), label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("OOD ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def fpr_at_tpr(tpr, fpr, target=0.95):
    idx = torch.searchsorted(tpr, torch.tensor(target, device=tpr.device), right=False)
    idx = min(int(idx.item()), len(fpr) - 1)
    return fpr[idx].item()


def main():
    args = parse_args()
    id_root = Path(args.id_root)
    ood_root = Path(args.ood_root)
    if not id_root.exists():
        raise FileNotFoundError(f"ID root not found: {id_root}")
    if not ood_root.exists():
        raise FileNotFoundError(f"OOD root not found: {ood_root}")

    val_tfms = transforms.Compose(
        [
            transforms.Resize(args.image_size + 32),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    id_ds = datasets.ImageFolder(args.id_root, transform=val_tfms)
    if args.ood_flat:
        ood_ds = FlatImageFolder(args.ood_root, transform=val_tfms)
    else:
        ood_ds = datasets.ImageFolder(args.ood_root, transform=val_tfms)
    id_loader = DataLoader(
        id_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    ood_loader = DataLoader(
        ood_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes=len(id_ds.classes)).to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    id_scores = compute_scores(model, id_loader, args.metric, args.temperature, device)
    ood_scores = compute_scores(model, ood_loader, args.metric, args.temperature, device)

    scores = torch.cat([id_scores, ood_scores], dim=0)
    labels = torch.cat([torch.ones_like(id_scores), torch.zeros_like(ood_scores)], dim=0)

    auc, tpr, fpr = roc_auc(scores, labels)
    fpr95 = fpr_at_tpr(tpr, fpr, target=0.95)

    print(f"metric={args.metric}")
    print(f"id_samples={len(id_scores)} ood_samples={len(ood_scores)}")
    print(f"AUROC={auc:.4f} FPR@95TPR={fpr95:.4f}")
    if args.plot:
        save_roc_plot(fpr, tpr, args.plot)
        print(f"Saved ROC curve to {args.plot}")


if __name__ == "__main__":
    main()
