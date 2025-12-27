#!/usr/bin/env python3
import argparse
import os
import random
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="SpaceNet training (DDP-ready).")
    parser.add_argument("--data-root", default="./img", help="Path to dataset root.")
    parser.add_argument("--model", default="convnext_tiny", help="torchvision model name.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    parser.add_argument("--no-class-weights", dest="class_weights", action="store_false")
    parser.set_defaults(class_weights=True)
    parser.add_argument("--out-dir", default="./checkpoints")
    return parser.parse_args()


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        distributed = False
    return distributed, rank, world_size, local_rank


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def compute_class_weights(dataset, device):
    counts = torch.zeros(len(dataset.classes), dtype=torch.long)
    for _, target in dataset.samples:
        counts[target] += 1
    counts = counts.clamp_min(1)
    weights = counts.sum() / (len(counts) * counts.float())
    return weights.to(device)


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank = init_distributed()
    set_seed(args.seed + rank)

    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize(args.image_size + 32),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    full_ds = datasets.ImageFolder(args.data_root, transform=train_tfms)
    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    split_gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=split_gen)
    val_ds.dataset = datasets.ImageFolder(args.data_root, transform=val_tfms)

    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes=len(full_ds.classes)).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    class_weights = None
    if args.class_weights:
        class_weights = compute_class_weights(full_ds, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        val_acc = val_correct / max(1, val_total)

        if rank == 0:
            print(
                f"[{epoch:03d}/{args.epochs}] "
                f"loss={train_loss:.4f} acc={train_acc:.3f} val_acc={val_acc:.3f}"
            )
            if epoch == args.epochs:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ckpt = {
                    "model": model.module.state_dict() if distributed else model.state_dict(),
                    "classes": full_ds.classes,
                    "args": vars(args),
                }
                torch.save(ckpt, os.path.join(args.out_dir, f"spacenet_{args.model}_{stamp}.pt"))

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
