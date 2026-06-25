
# RoadHeightFormer (RHF)

RoadHeightFormer (RHF) is a monocular road surface elevation estimation framework that builds on the Bird's Eye View pipeline introduced by RoadBEV. It replaces the EfficientNet encoder with a frozen DINOv2 ViT-S/14 backbone, adds a patch2feature upsampler to recover spatial resolution, and trains with a composite loss combining L1, multi-scale gradient, and surface-normal terms. Evaluated on the CARDSet dataset, RHF achieves approximately **30% improvement in absolute error over RoadBEV**.

## Dataset — CARDSet

CARDSet is a proprietary CARIAD dataset of front-camera images with LiDAR-aggregated road surface elevation ground truth stored as BEV height maps.

**Directory layout expected:**
```
/data/T7/cariad dataset/          # raw images (read by the dataloader)
/data/rhf/
  ├── val_small_dataset_thesis.txt          # val split file (one timestamp per line)
  ├── val_preprocessed_small_data_thesis/  # preprocessed val pkls (data_item_XXXXXX.pkl.gz)
  └── checkpoints/                          # saved model checkpoints
```

**Preprocessing** — run once to generate the `.pkl.gz` files the dataloader expects:
```bash
python preprocess_gt.py --save_dir /data/rhf/preprocessed/ --dataset train
python preprocess_gt.py --save_dir /data/rhf/preprocessed/ --dataset val
```

Each `.pkl.gz` contains the cropped image tensor, BEV elevation map (`ele_gt`), validity mask, voxel UV projection indices, camera intrinsics, and pose.

---

## Train

All hyperparameters are controlled via a YAML config. The baseline RHF config is `configs/config_freeze_baseline.yaml`.

```bash
python train.py --config configs/config_freeze_baseline.yaml
```

Key flags you may want to override on the command line:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 30 | Number of training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-4 | Peak learning rate |
| `--logdir` | `/data/rhf/checkpoints/` | Where to save checkpoints |
| `--loadckpt` | None | Resume from a checkpoint |
| `--backbone` | `DINOv2_fb` | Encoder: `DINOv2_fb`, `efficientnet`, `DA3-SMALL` |
| `--train_encoder` | off | Unfreeze the backbone during training |
| `--preprocessed` | off | Use preprocessed `.pkl.gz` files (strongly recommended) |

**Resume a run:**
```bash
python train.py --config configs/config_freeze_baseline.yaml --load_pt /data/rhf/checkpoints/<run>/checkpoint_epoch05_001234.pt
```

---

## Test / Evaluate

**Single checkpoint on the CARDSet val split:**
```bash
python test.py --config configs/config_freeze_baseline.yaml \
               --loadckpt /data/rhf/checkpoints/<run>/final_<name>.pt
```

**Batch-evaluate all final checkpoints and print a results table:**
```bash
python eval_all_finals.py
```

Results on the CARDSet val split (764 samples):

| Model | AbsErr (cm) | RMSE (cm) | LE90 (cm) | GradErr | >0.5cm (%) |
|---|---:|---:|---:|---:|---:|
| **RHF baseline** | **3.118** | **3.525** | **4.991** | **0.2501** | 83.5 |
| RHF clamp GT | 3.125 | 3.534 | 5.006 | 0.2502 | 83.6 |
| RHF L1 only | 3.310 | 3.757 | 5.351 | 0.2664 | 84.1 |
| RHF MSE only | 3.102 | 3.513 | 5.000 | 0.2700 | 83.7 |
| RHF crop-to-road | 6.038 | 6.934 | 10.558 | 0.2600 | 94.1 |
| RHF dino upsampler | 3.335 | 3.810 | 5.552 | 0.2675 | 86.5 |
| RHF layers [11,11,11,11] | 3.112 | 3.530 | 5.030 | 0.2496 | 83.7 |
| RHF trainable encoder | 3.505 | 3.878 | 5.358 | 0.2473 | 84.7 |
| RHF classification head | 3.440 | 3.946 | 5.611 | 0.2581 | 83.8 |
| RHF EfficientNet backbone | 6.622 | 7.544 | 11.201 | 0.4383 | 94.7 |
| RHF DepthAnything3 backbone | 3.695 | 4.162 | 5.980 | 0.2665 | 87.8 |
| RoadBEV (baseline) | 4.836 | 5.495 | 8.005 | 0.3225 | 90.2 |

---

## Key Files & Folders

| Path | Description |
|---|---|
| `train.py` | Main training loop — loss, optimiser, checkpointing, and logging |
| `models/model_dinov2_fb.py` | RHF model: frozen DINOv2 encoder + patch2feature upsampler + BEV elevation head |
| `models/structural_losses.py` | Composite loss: L1/MSE + multi-scale gradient + surface-normal cosine |
| `cardset/dataset.py` | CARDSet dataloader — preprocessed pkls, voxel UV projection indices, augmentation |
| `utils/metric.py` | Metrics: AbsErr, RMSE, LE90, GradErr, ratio thresholds (0.1 / 0.5 / 1.0 cm) |
| `eval_all_finals.py` | Batch-evaluates every `final_*.pt` checkpoint on the CARDSet val split |
| `configs/config_freeze_baseline.yaml` | Baseline RHF config: frozen DINOv2, composite loss, regression head |
| `models/ele_head.py` | BEV elevation prediction head (classification or regression) |
| `utils/normals.py` | Surface-normal computation from 3-D point clouds (used by structural loss) |
| `models/model.py` | RoadBEV / DA3 baseline model (EfficientNet / DepthAnything3 backbone) |

## Evaluation Scripts

All eval scripts share the same structure: define a `RUNS` list of `(label, checkpoint, config)` tuples, build the model from config, run inference over the val split using `utils/metric.py`, and print a markdown table of results (also saved as JSON).

| Script | What it compares |
|---|---|
| `eval_all_finals.py` | Every `final_*.pt` checkpoint on the CARDSet val split — main ablation table |
| `eval_rhf_vs_da3_per_sample.py` | RHF baseline vs DA3-SMALL per sample (with scale+shift alignment for DA3) |
| `eval_da3_aligned_full.py` | DA3-SMALL with per-image scale+shift alignment on the full val split |
| `eval_da3metric_large.py` | DA3-METRIC-LARGE (metric depth, no alignment needed) on the CARDSet val split |
| `eval_rsrd_baseline_vs_roadbev.py` | RHF baseline vs RoadBEV on the RSRD-dense test split |
