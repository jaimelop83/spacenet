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

OOD evaluation with a flat folder and ROC plot:
```
python evaluate_ood.py --id-root ./img --ood-root ./ood_flat --ood-flat \
  --checkpoint ./checkpoints/spacenet_convnext_tiny_<timestamp>.pt \
  --metric energy --plot ./logs/ood_roc.png
```

Quick OOD eval using latest checkpoint:
```
bash scripts/run_ood_eval.sh ./img ./ood_flat --ood-flat --metric energy --plot ./logs/ood_roc.png
```

OOD per-image scores (optional threshold to label ID/OOD):
```
python evaluate_ood.py --id-root ./img --ood-root ./ood_flat --ood-flat \
  --checkpoint ./checkpoints/spacenet_convnext_tiny_<timestamp>.pt \
  --metric max_softmax --threshold 0.5 --out-csv ./logs/ood_scores.csv
```

Auto-threshold by target TPR (uses ID scores):
```
python evaluate_ood.py --id-root ./img --ood-root ./ood_flat --ood-flat \
  --checkpoint ./checkpoints/spacenet_convnext_tiny_<timestamp>.pt \
  --metric max_softmax --auto-threshold-tpr 0.95 --out-csv ./logs/ood_scores.csv
```

OOD per-image scores with DDP (both GPUs):
```
torchrun --nproc_per_node=2 evaluate_ood.py --id-root ./img --ood-root /mnt/personal_drive/jwst/previews --ood-flat \
  --checkpoint ./checkpoints/spacenet_convnext_tiny_<timestamp>.pt \
  --metric max_softmax --auto-threshold-tpr 0.95 --out-csv ./logs/ood_scores.csv
```

JWST preview download (to UNAS):
```
python scripts/download_jwst_previews.py --out-dir /mnt/personal_drive/jwst/previews \
  --max-files 1000 --instruments NIRCAM MIRI
```

JWST FITS download (to UNAS):
```
python scripts/download_jwst_fits.py --out-dir /mnt/personal_drive/jwst/fits \
  --max-files 500 --instruments NIRCAM MIRI
```

JWST progress and log rotation:
```
bash scripts/check_jwst_progress.sh
bash scripts/rotate_jwst_logs.sh 14
```


Visual report (confusion matrix, sample grid, training curves):
```
python report_visuals.py --data-root ./img \
  --csv-log ./logs/train_metrics.csv --out-dir ./reports
```
