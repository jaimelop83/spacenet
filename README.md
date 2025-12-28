# SpaceNet Training

Dataset layout:
```
spacenet/
  img/
    asteroid/
    black_hole/
    comet/
    constellation/
    galaxy/
    nebula/
    planet/
    star/
```

Single GPU (or CPU):
```
python train.py --data-root ./img --epochs 30 --batch-size 64 --amp
```

Both GPUs (DDP):
```
torchrun --nproc_per_node=2 train.py --data-root ./img --epochs 30 --batch-size 64 --amp
```

Notes:
- Default model is `convnext_tiny` from torchvision; swap with `--model resnet50` or similar.
- Checkpoints are saved to `./checkpoints` at the final epoch.
- Use `--no-class-weights` to disable class-balanced loss.
- Logs: CSV at `./logs/` and TensorBoard at `./runs/`.
- Augmentation ablation: `--aug-strength light|default|heavy`.

SimCLR pretraining (DDP):
```
torchrun --nproc_per_node=2 pretrain_simclr.py --data-root ./img --epochs 200 --batch-size 128 --amp
```

Resume SimCLR (DDP) and save every 10 epochs:
```
torchrun --nproc_per_node=2 pretrain_simclr.py --data-root ./img --epochs 200 --batch-size 128 --amp \
  --resume ./checkpoints/simclr_convnext_tiny_<timestamp>.pt --save-every 10
```

Fine-tune from SimCLR encoder:
```
torchrun --nproc_per_node=2 train.py --data-root ./img --epochs 30 --batch-size 64 --amp \
  --pretrained-encoder ./checkpoints/simclr_convnext_tiny_<timestamp>.pt
```

Resume fine-tune (DDP) and save every 5 epochs:
```
torchrun --nproc_per_node=2 train.py --data-root ./img --epochs 30 --batch-size 64 --amp \
  --resume ./checkpoints/spacenet_convnext_tiny_<timestamp>.pt --save-every 5
```

Augmentation ablation (light/default/heavy):
```
bash scripts/run_ablation.sh ./img ./checkpoints/simclr_convnext_tiny_<timestamp>.pt 30 64
```

OOD evaluation (requires a separate OOD folder with class subfolders):
```
python evaluate_ood.py --id-root ./img --ood-root ./ood_img \
  --checkpoint ./checkpoints/spacenet_convnext_tiny_<timestamp>.pt --metric max_softmax
```
