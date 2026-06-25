"""Evaluate DA3METRIC-LARGE (metric monocular depth) on the CARDSet val split.
No scale/shift alignment is needed — the network output is converted to metres
directly via `metric_depth = focal * net_output / 300`  (per the DA3 README).

For each val sample:
  1. Run DA3METRIC-LARGE on the original RGB resized to 560×560.
  2. Convert net output to metric depth using fx, fy from the sample's K.
  3. Back-project to camera-frame points; bin into the BEV grid with the
     widened y_crop = (-0.5, 3.0) so that downhill / pitched scenes don't get
     clipped.
  4. Compute per-sample MAE / RMSE vs the dataset's preprocessed GT.

Also saves heightmap quadviews (RGB | GT | RHF | DA3-LARGE-METRIC) for the
same 10 hand-picked val indices.

Outputs:
  eval_da3metric_large_per_sample.json
  preds_da3metric_large/{idx:03d}_{tag}/{predictions.npz, quadview.png, rgb.jpg}
"""
import gzip
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/f9ql00v/RoadHeightformer")

from utils.config import create_parser, load_config, update_args_with_config
from cardset.dataset import CARDSetDataset
from models.model_dinov2_fb import Elevation as ElevationDinoV2FB
from run_da3_inference import pointcloud_to_height_map, RESIZE_WH, DA3_SRC

if DA3_SRC not in sys.path:
    sys.path.insert(0, DA3_SRC)
from depth_anything_3.api import DepthAnything3  # noqa: E402


CARDSET_ROOT     = "/data/T7/cariad dataset"
VAL_SPLIT        = "/data/rhf/val_small_dataset_thesis.txt"
PREPROCESSED_DIR = "/data/rhf/val_preprocessed_small_data_thesis"
DA3_NAME         = "depth-anything/DA3METRIC-LARGE"
WIDE_ROI_Y       = (-0.5, 3.0)

RHF_CKPT = "/data/rhf/checkpoints/RHF_compositeloss_dinov2_rhf_baseline/final_RHF_compositeloss_dinov2_rhf_baseline_epoch30_007860.pt"
RHF_CFG  = "configs/config_freeze_baseline.yaml"

OUT_JSON = "eval_da3metric_large_per_sample.json"
PRED_ROOT = "preds_da3metric_large"

PICKS = [
    (760, "best_RHF"),
    (645, "best_RHF"),
    (196, "best_RHF"),
    (726, "median"),
    (483, "median"),
    (653, "worst_RHF"),
    (162, "worst_RHF"),
    (164, "worst_RHF"),
    (181, "biggest_gain"),
    (131, "biggest_gain"),
]


def per_sample_errors(ele_pred, ele_gt, mask):
    """Same metric as eval_rhf_vs_da3_per_sample.py: MAE/RMSE on full GT range."""
    m = mask.astype(bool)
    if not m.any():
        return float("nan"), float("nan"), 0
    err = (ele_gt[m] - ele_pred[m]).astype(np.float64)
    return float(np.abs(err).mean()), float(np.sqrt(np.mean(err ** 2))), int(m.sum())


def da3_metric_to_depth(net_output, fx, fy):
    """Per the DA3 README: metric_depth = focal * net_output / 300.
    focal = mean(fx, fy) in pixels of the model's INPUT image (560x560)."""
    focal = 0.5 * (fx + fy)
    return focal * net_output / 300.0


def save_quadview(out_png, rgb_path, ele_gt, mask_gt, rhf_pred, rhf_mask,
                  da3_pred, da3_mask, title):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    try:
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        axs[0].imshow(rgb)
    except Exception:
        axs[0].text(0.5, 0.5, "RGB unavailable", ha="center", va="center")
    axs[0].set_title("RGB input"); axs[0].set_axis_off()

    gt_valid = ele_gt[mask_gt]
    if gt_valid.size > 0:
        vmin = float(np.nanmin(gt_valid)); vmax = float(np.nanmax(gt_valid))
        if vmax - vmin < 1.0:
            mid = 0.5 * (vmin + vmax); vmin, vmax = mid - 0.5, mid + 0.5
    else:
        vmin, vmax = -20.0, 20.0
    cmap = plt.get_cmap("plasma").copy(); cmap.set_bad("black")

    def panel(ax, ele, m, sub):
        masked = np.ma.masked_where(~m, ele)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest", aspect="auto")
        ax.set_title(sub, fontsize=11)
        ax.set_xlabel("X grid"); ax.set_ylabel("Z grid")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("elevation (cm)")

    panel(axs[1], ele_gt,   mask_gt,            "GT elevation")
    panel(axs[2], rhf_pred, rhf_mask & mask_gt, "RHF baseline")
    panel(axs[3], da3_pred, da3_mask & mask_gt, "DA3METRIC-LARGE (no align)")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close(fig)


@torch.no_grad()
def main():
    torch.backends.cudnn.benchmark = True
    if os.path.isdir(PRED_ROOT):
        shutil.rmtree(PRED_ROOT)
    os.makedirs(PRED_ROOT, exist_ok=True)

    # --- RHF (for quadviews) ---
    parser = create_parser()
    args = parser.parse_args([])
    args = update_args_with_config(args, load_config(RHF_CFG))
    args.down_scale = 4

    print("[rhf] building dataset ...")
    test_set = CARDSetDataset(
        root_dir=CARDSET_ROOT, split_file=VAL_SPLIT, mode="test",
        down_scale=args.down_scale, preprocessed_data=args.preprocessed,
        augmentation=False, clamp_gt=args.clamp_gt, crop_to_road=args.crop_to_road,
    )
    test_loader = DataLoader(test_set, 1, shuffle=False, num_workers=2, drop_last=False, pin_memory=False)
    print(f"[rhf] val size = {len(test_set)}")

    rhf = ElevationDinoV2FB(
        stereo=False,
        num_grids=[test_set.num_grids_x, test_set.num_grids_y, test_set.num_grids_z],
        ele_range=test_set.y_range, cla_res=args.cla_res, regression=args.regression,
        backbone=args.backbone, normalize=args.normalize, pred_dim=args.pred_head_dim,
        train_encoder=args.train_encoder, dinov2_layers=tuple(args.dinov2_layers),
        upsampler_kind=args.upsampler_kind,
    ).cuda().eval()
    ckpt = torch.load(RHF_CKPT, map_location="cuda")
    rhf.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)

    # --- DA3METRIC-LARGE ---
    print(f"[da3] loading {DA3_NAME} ...")
    da3 = DepthAnything3.from_pretrained(DA3_NAME).to(device="cuda").eval()

    picks_set = {p[0]: p[1] for p in PICKS}

    per_sample = []
    sum_mae = sum_rmse = 0.0
    n_count = n_skip = 0
    t0 = time.time()

    for i, sample in enumerate(test_loader):
        try:
            imgs_left, ele_gt_t, ele_mask_t, proj_index_left, _ = sample
        except Exception:
            n_skip += 1
            continue
        ele_gt_np = ele_gt_t.squeeze(0).numpy().astype(np.float32)
        mask_gt_np = ele_mask_t.squeeze(0).numpy().astype(bool)

        pkl_path = os.path.join(PREPROCESSED_DIR, f"data_item_{i:06d}.pkl.gz")
        try:
            with gzip.open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            n_skip += 1
            continue

        img_path = data["path"]
        img_pil = Image.open(img_path).convert("RGB").resize(RESIZE_WH)
        pred = da3.inference([img_pil], process_res=RESIZE_WH[0])
        net_output = pred.depth[0].astype(np.float32)

        intr = np.asarray(data["intrinsics"], dtype=np.float32)
        extr_c2w = np.asarray(data["extrinsics"], dtype=np.float32)
        ground_normal = data["ground_normal"]
        camera_height = float(data["camera_height"])

        # Metric conversion (no s,t alignment needed)
        fx, fy = float(intr[0, 0]), float(intr[1, 1])
        cx, cy = float(intr[0, 2]), float(intr[1, 2])
        depth_metric = da3_metric_to_depth(net_output, fx, fy)

        H, W = depth_metric.shape
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        zc = depth_metric
        xc = (uu - cx) * zc / fx
        yc = (vv - cy) * zc / fy
        pts_cam = np.stack([xc, yc, zc], axis=-1).reshape(-1, 3)
        valid = (zc.reshape(-1) > 0) & np.isfinite(zc.reshape(-1))
        pts_cam = pts_cam[valid]

        da3_pred_np, da3_mask_np = pointcloud_to_height_map(
            points_cam=pts_cam, ground_normal_world=ground_normal,
            R_c2w=extr_c2w[:3, :3], camera_height=camera_height,
            y_crop=WIDE_ROI_Y,
        )

        m_combined = da3_mask_np & mask_gt_np
        mae, rmse, n_cells = per_sample_errors(da3_pred_np, ele_gt_np, m_combined)

        if np.isfinite(mae):
            sum_mae += mae; sum_rmse += rmse; n_count += 1
        else:
            n_skip += 1

        per_sample.append({
            "idx": i, "img_path": img_path,
            "da3_mae": mae, "da3_rmse": rmse, "n_cells": n_cells,
            "depth_metric_median_m": float(np.median(depth_metric)),
        })

        # Save quadview for picked samples
        if i in picks_set:
            imgs_left_c = imgs_left.cuda()
            proj_idx_c = proj_index_left.cuda()
            rhf_pred = rhf(imgs_left_c, proj_idx_c).squeeze().cpu().numpy().astype(np.float32)
            rhf_mask = np.ones_like(mask_gt_np, dtype=bool)

            def mae_rmse(p, m):
                mm = m & mask_gt_np
                if not mm.any(): return float("nan"), float("nan")
                err = (ele_gt_np - p)[mm]
                return float(np.abs(err).mean()), float(np.sqrt(np.mean(err ** 2)))
            rhf_mae, rhf_rmse = mae_rmse(rhf_pred, rhf_mask)

            tag = picks_set[i]
            out_dir = os.path.join(PRED_ROOT, f"{i:03d}_{tag}")
            os.makedirs(out_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(out_dir, "predictions.npz"),
                idx=i, tag=tag, img_path=img_path,
                ele_gt=ele_gt_np, mask_gt=mask_gt_np,
                rhf_pred=rhf_pred,
                da3_pred=da3_pred_np, da3_mask=da3_mask_np,
                depth_metric=depth_metric,
                rhf_mae=rhf_mae, rhf_rmse=rhf_rmse,
                da3_mae=mae, da3_rmse=rmse,
            )
            title = (f"idx={i}  [{tag}]   RHF MAE/RMSE={rhf_mae:.2f}/{rhf_rmse:.2f} cm   "
                     f"DA3METRIC-LARGE MAE/RMSE={mae:.2f}/{rmse:.2f} cm")
            save_quadview(os.path.join(out_dir, "quadview.png"),
                          img_path, ele_gt_np, mask_gt_np,
                          rhf_pred, rhf_mask, da3_pred_np, da3_mask_np, title)
            try:
                shutil.copy(img_path, os.path.join(out_dir, "rgb.jpg"))
            except Exception:
                pass

        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(test_loader)}  elapsed={time.time()-t0:.1f}s  "
                  f"avg_mae={sum_mae/max(n_count,1):.2f}")

    avg_mae = sum_mae / max(n_count, 1)
    avg_rmse = sum_rmse / max(n_count, 1)
    summary = {
        "model": DA3_NAME,
        "roi_y": list(WIDE_ROI_Y),
        "n_samples": n_count,
        "n_skipped": n_skip,
        "abs_err_cm": avg_mae,
        "rmse_cm": avg_rmse,
        "per_sample": per_sample,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] elapsed={time.time()-t0:.1f}s   n={n_count}   skipped={n_skip}")
    print(f"\n  DA3METRIC-LARGE   MAE = {avg_mae:.3f} cm   RMSE = {avg_rmse:.3f} cm")
    print(f"\nResults → {OUT_JSON}")
    print(f"Per-sample preds → {PRED_ROOT}/")

    # Picks summary
    by_idx = {s["idx"]: s for s in per_sample}
    print("\nHand-picked samples (DA3METRIC-LARGE only):")
    print("| idx | tag           | MAE (cm) | RMSE (cm) |")
    print("|----:|---------------|---------:|----------:|")
    for idx, tag in PICKS:
        r = by_idx.get(idx)
        if r is None or not np.isfinite(r.get("da3_mae", float("nan"))):
            print(f"| {idx:3d} | {tag:<13} |        — |         — |")
        else:
            print(f"| {idx:3d} | {tag:<13} | {r['da3_mae']:8.3f} | {r['da3_rmse']:9.3f} |")


if __name__ == "__main__":
    main()
