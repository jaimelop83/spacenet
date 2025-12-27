#!/usr/bin/env python3
import argparse
import csv
import os
import random
from datetime import datetime

import torch
import torch.distributed as dist
from PIL import UnidentifiedImageError, Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="SimCLR pretraining (DDP-ready).")
    parser.add_argument("--data-root", default="./img", help="Path to dataset root.")
    parser.add_argument("--model", default="convnext_tiny", help="torchvision model name.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    parser.add_argument("--out-dir", default="./checkpoints")
    parser.add_argument("--log-dir", default="./runs")
    parser.add_argument("--csv-log", default="./logs/pretrain_metrics.csv")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
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


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


def build_backbone(name, image_size, device):
    if not hasattr(models, name):
        raise ValueError(f"Unknown torchvision model: {name}")
    model_fn = getattr(models, name)
    model = model_fn(weights=None)
    if hasattr(model, "classifier"):
        model.classifier = torch.nn.Identity()
    elif hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    else:
        raise ValueError("Unsupported model head; add a head mapping here.")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        feat = model(dummy)
        if feat.dim() > 2:
            feat = feat.mean(dim=(2, 3))
        feat_dim = feat.shape[-1]
    model.train()
    return model, feat_dim


class SimCLR(torch.nn.Module):
    def __init__(self, backbone, feat_dim, proj_dim=128):
        super().__init__()
        self.encoder = backbone
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, feat_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        feats = self.encoder(x)
        if feats.dim() > 2:
            feats = feats.mean(dim=(2, 3))
        proj = self.projector(feats)
        return proj


def nt_xent_loss(z, temperature):
    z = torch.nn.functional.normalize(z, dim=1).float()
    local_batch = z.size(0)
    if local_batch % 2 != 0:
        raise ValueError("SimCLR batch must be even (two views per sample).")

    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if world_size > 1:
        z_list = [torch.zeros_like(z) for _ in range(world_size)]
        dist.all_gather(z_list, z)
        world_z = torch.cat(z_list, dim=0)
    else:
        world_z = z

    sim = torch.matmul(z, world_z.t()) / temperature
    sim = sim.float()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    global_start = rank * local_batch
    global_indices = torch.arange(local_batch, device=z.device) + global_start
    sim = sim.scatter(1, global_indices.unsqueeze(1), -1e9)

    half = local_batch // 2
    pos = torch.arange(local_batch, device=z.device)
    pos = torch.where(pos < half, pos + half, pos - half)
    pos = pos + global_start

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -log_prob[torch.arange(local_batch, device=z.device), pos]
    return loss.mean()


def filter_bad_samples(samples):
    good = []
    bad = []
    for path, target in samples:
        try:
            with Image.open(path) as img:
                img.verify()
            good.append((path, target))
        except (UnidentifiedImageError, OSError):
            bad.append(path)
    return good, bad


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank = init_distributed()
    set_seed(args.seed + rank)

    base_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    dataset = datasets.ImageFolder(args.data_root, transform=TwoCropsTransform(base_transform))
    dataset.samples, bad = filter_bad_samples(dataset.samples)
    dataset.imgs = dataset.samples
    if rank == 0 and bad:
        print(f"Skipped {len(bad)} unreadable images during pretrain.")
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, feat_dim = build_backbone(args.model, args.image_size, device)
    model = SimCLR(backbone, feat_dim).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model_ref = model.module if distributed else model
        model_ref.encoder.load_state_dict(ckpt["encoder"])
        model_ref.projector.load_state_dict(ckpt["projector"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and args.amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if rank == 0:
            print(f"Resuming from {args.resume} at epoch {start_epoch}")

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(os.path.dirname(args.csv_log), exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)
        csv_exists = os.path.exists(args.csv_log)
        csv_file = open(args.csv_log, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if not csv_exists:
            csv_writer.writerow(["epoch", "loss", "lr"])
    else:
        writer = None
        csv_file = None
        csv_writer = None

    for epoch in range(start_epoch, args.epochs + 1):
        if distributed:
            sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        total = 0

        for (x1, x2), _ in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            images = torch.cat([x1, x2], dim=0)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                z = model(images)
                loss = nt_xent_loss(z, args.temperature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

        if rank == 0:
            avg_loss = running_loss / max(1, total)
            lr = optimizer.param_groups[0]["lr"]
            print(f"[{epoch:03d}/{args.epochs}] loss={avg_loss:.4f}")
            csv_writer.writerow([epoch, f"{avg_loss:.6f}", lr])
            csv_file.flush()
            writer.add_scalar("train/loss", avg_loss, epoch)
            writer.add_scalar("train/lr", lr, epoch)
            if epoch % args.save_every == 0 or epoch == args.epochs:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_ref = model.module if distributed else model
                ckpt = {
                    "encoder": model_ref.encoder.state_dict(),
                    "projector": model_ref.projector.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if args.amp else None,
                    "epoch": epoch,
                    "args": vars(args),
                }
                torch.save(ckpt, os.path.join(args.out_dir, f"simclr_{args.model}_{stamp}.pt"))

    if distributed:
        dist.destroy_process_group()
    if rank == 0:
        writer.close()
        csv_file.close()


if __name__ == "__main__":
    main()
