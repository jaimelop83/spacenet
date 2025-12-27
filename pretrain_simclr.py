#!/usr/bin/env python3
import argparse
import os
import random
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, models, transforms


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
        proj = self.projector(feats)
        return proj


def nt_xent_loss(z, temperature):
    z = torch.nn.functional.normalize(z, dim=1)
    batch_size = z.size(0) // 2
    sim = torch.matmul(z, z.t()) / temperature
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -9e15)

    pos_indices = torch.arange(batch_size, device=z.device)
    pos_indices = torch.cat([pos_indices + batch_size, pos_indices])
    logits = sim
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = -log_prob[torch.arange(2 * batch_size, device=z.device), pos_indices]
    return loss.mean()


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

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
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
            print(f"[{epoch:03d}/{args.epochs}] loss={avg_loss:.4f}")
            if epoch == args.epochs:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ckpt = {
                    "encoder": model.module.encoder.state_dict() if distributed else model.encoder.state_dict(),
                    "projector": model.module.projector.state_dict()
                    if distributed
                    else model.projector.state_dict(),
                    "args": vars(args),
                }
                torch.save(ckpt, os.path.join(args.out_dir, f"simclr_{args.model}_{stamp}.pt"))

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
