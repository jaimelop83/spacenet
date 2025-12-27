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
