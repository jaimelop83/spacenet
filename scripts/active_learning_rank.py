#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Rank unlabeled images by uncertainty.")
    parser.add_argument("--unlabeled-root", required=True, help="Folder of unlabeled images.")
    parser.add_argument("--checkpoint", required=True, help="Classifier checkpoint path.")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


class FlatImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = self._collect_images(self.root)
        self.samples, self.skipped = self._filter_bad_paths(self.samples)

    @staticmethod
    def _collect_images(root):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        files = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                files.append(path)
        return sorted(files)

    @staticmethod
    def _filter_bad_paths(paths):
        good = []
        skipped = 0
        for path in paths:
            try:
                with Image.open(path) as img:
                    img.verify()
                good.append(path)
            except (UnidentifiedImageError, OSError):
                skipped += 1
                continue
        return good, skipped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(path)


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


def main():
    args = parse_args()

    val_tfms = transforms.Compose(
        [
            transforms.Resize(args.image_size + 32),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    dataset = FlatImageFolder(args.unlabeled_root, transform=val_tfms)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    classes = ckpt.get("classes", None) if isinstance(ckpt, dict) else None
    num_classes = len(classes) if classes else 8
    model = build_model(args.model, num_classes=num_classes).to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    rows = []
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
            for p, c, e, pred_idx in zip(paths, conf.cpu(), entropy.cpu(), pred.cpu()):
                rows.append((str(p), int(pred_idx), float(c), float(e)))

    rows.sort(key=lambda r: (-r[3], r[2]))
    rows = rows[: args.top_n]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "pred_class", "confidence", "entropy", "label"])
        for path, pred_idx, conf, ent in rows:
            writer.writerow([path, pred_idx, f"{conf:.6f}", f"{ent:.6f}", ""])

    print(f"wrote {out_path}")
    if dataset.skipped:
        print(f"skipped_unreadable={dataset.skipped}")


if __name__ == "__main__":
    main()
