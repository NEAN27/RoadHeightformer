"""Combined per-sample evaluation: RHF baseline vs Depth-Anything-3 (DA3-SMALL).

For every sample in the CARDSet val split it:
  1. runs the RHF baseline (DINOv2_fb encoder + patch2feature upsampler) and
     records its per-sample MAE & RMSE against the BEV ground-truth elevation map.
  2. runs DA3-SMALL on the same RGB image, aligns its monocular depth with a
     per-image scale+shift (using the LiDAR-aggregated GT when available, falling
     back to the BEV-derived synthetic depths), back-projects it into the camera
     frame, bins the result into the BEV grid (`pointcloud_to_height_map`) and
     records its per-sample MAE & RMSE against the same GT.

Outputs:
  * overall metrics for both models (mean of per-sample abs_err / RMSE),
  * a JSON file with the per-sample errors keyed by dataset index,
  * a printed table of 10 hand-picked samples (best / median / worst, plus a few
    where RHF beats DA3 by the largest margin) for the reference letter.
"""
import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/f9ql00v/RoadHeightformer")

from utils.config import create_parser, load_config, update_args_with_config
from cardset.dataset import CARDSetDataset
from models.model_dinov2_fb import Elevation as ElevationDinoV2FB
from run_da3_inference import (
    align_depth_scale_shift,
    align_depth_scale_shift_from_aggdepth,
    pointcloud_to_height_map,
    RESIZE_WH,
    DA3_SRC,
)

if DA3_SRC not in sys.path:
    sys.path.insert(0, DA3_SRC)
from depth_anything_3.api import DepthAnything3  # noqa: E402


CARDSET_ROOT     = "/data/T7/cariad dataset"
VAL_SPLIT        = "/data/rhf/val_small_dataset_thesis.txt"
PREPROCESSED_DIR = "/data/rhf/val_preprocessed_small_data_thesis"

RHF_CKPT = "/data/rhf/checkpoints/RHF_compositeloss_dinov2_rhf_baseline/final_RHF_compositeloss_dinov2_rhf_baseline_epoch30_007860.pt"
RHF_CFG  = "configs/config_freeze_baseline.yaml"
DA3_NAME = "depth-anything/DA3-SMALL"

OUTPUT_JSON = "eval_rhf_vs_da3_per_sample.json"
ELE_RANGE_M = 0.2  # ±20 cm = the BEV elevation half-range used for clamping in the GT


def per_sample_errors(ele_pred, ele_gt, mask):
    """abs_err mean (cm) and RMSE (cm) over the masked region — uses the full GT
    range (no ±ele_range clamp). Returns (np.nan, np.nan, 0) if empty."""
    m = mask.bool()
    n = int(m.sum().item())
    if n == 0:
        return float("nan"), float("nan"), 0
    err = (ele_gt[m] - ele_pred[m]).float()
    mae = err.abs().mean().item()
    rmse = (err.pow(2).mean().sqrt()).item()
    return float(mae), float(rmse), n


@torch.no_grad()
def main():
    torch.backends.cudnn.benchmark = True

    # ---------------- RHF baseline ----------------
    parser = create_parser()
    args = parser.parse_args([])
    cfg = load_config(RHF_CFG)
    args = update_args_with_config(args, cfg)
    args.down_scale = 4

    print(f"[rhf]   building CARDSetDataset val split (preprocessed={args.preprocessed}) ...")
    test_set = CARDSetDataset(
        root_dir=CARDSET_ROOT,
        split_file=VAL_SPLIT,
        mode='test',
        down_scale=args.down_scale,
        preprocessed_data=args.preprocessed,
        augmentation=False,
        clamp_gt=args.clamp_gt,
        crop_to_road=args.crop_to_road,
    )
    print(f"[rhf]   val set size = {len(test_set)}")
    test_loader = DataLoader(test_set, 1, shuffle=False, num_workers=2, drop_last=False, pin_memory=False)

    ele_range = test_set.y_range
    num_grids = [test_set.num_grids_x, test_set.num_grids_y, test_set.num_grids_z]

    print("[rhf]   loading checkpoint ...")
    rhf = ElevationDinoV2FB(
        stereo=False,
        num_grids=num_grids,
        ele_range=ele_range,
        cla_res=args.cla_res,
        regression=args.regression,
        backbone=args.backbone,
        normalize=args.normalize,
        pred_dim=args.pred_head_dim,
        train_encoder=args.train_encoder,
        dinov2_layers=tuple(args.dinov2_layers),
        upsampler_kind=args.upsampler_kind,
    ).cuda().eval()
    ckpt = torch.load(RHF_CKPT, map_location="cuda")
    rhf.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=True)

    # ---------------- DA3 ----------------
    print(f"[da3]   loading {DA3_NAME} ...")
    da3 = DepthAnything3.from_pretrained(DA3_NAME).to(device="cuda").eval()

    # ---------------- loop ----------------
    per_sample = []           # list of dicts: {idx, path, rhf_mae, rhf_rmse, da3_mae, da3_rmse, n_valid}
    rhf_mae_sum = rhf_rmse_sum = 0.0
    da3_mae_sum = da3_rmse_sum = 0.0
    n_rhf = n_da3 = 0
    n_skip_rhf = n_skip_da3 = 0
    t0 = time.time()

    for i, sample in enumerate(test_loader):
        try:
            imgs_left, ele_gt, ele_mask, proj_index_left, _ = sample
        except Exception:
            n_skip_rhf += 1
            n_skip_da3 += 1
            continue

        imgs_left = imgs_left.cuda()
        ele_gt_t = ele_gt.cuda()
        ele_mask_t = ele_mask.cuda()
        proj_index_left = proj_index_left.cuda()

        # ---- RHF forward ----
        rhf_pred = rhf(imgs_left, proj_index_left).squeeze(0)
        gt_2d = ele_gt_t.squeeze(0)
        mask_2d = ele_mask_t.squeeze(0)
        rhf_mae, rhf_rmse, n_rhf_cells = per_sample_errors(rhf_pred, gt_2d, mask_2d)

        # ---- DA3 forward ----
        pkl_path = os.path.join(PREPROCESSED_DIR, f"data_item_{i:06d}.pkl.gz")
        da3_mae = float("nan")
        da3_rmse = float("nan")
        img_path_str = None
        try:
            with gzip.open(pkl_path, "rb") as f:
                data = pickle.load(f)
            img_path_str = data["path"]
            img_pil = Image.open(img_path_str).convert("RGB").resize(RESIZE_WH)

            pred = da3.inference([img_pil], process_res=RESIZE_WH[0])
            depth = pred.depth[0].astype(np.float32)

            extr_c2w = np.asarray(data["extrinsics"], dtype=np.float32)
            intr = np.asarray(data["intrinsics"], dtype=np.float32)
            ground_normal = data["ground_normal"]
            camera_height = float(data["camera_height"])
            mask_gt_np = np.asarray(data["mask"]) > 0
            ele_gt_np = np.asarray(data["ele_gt"], dtype=np.float32)

            pts_cam_gt = None
            dpath = data.get("depth_path")
            if dpath and os.path.exists(dpath):
                try:
                    pts_cam_gt = np.load(dpath)["pts_cam"]
                except Exception:
                    pts_cam_gt = None

            if pts_cam_gt is not None:
                s, t, _ = align_depth_scale_shift_from_aggdepth(
                    depth_pred=depth, pts_cam=pts_cam_gt, intrinsics=intr,
                    ground_normal_world=ground_normal, R_c2w=extr_c2w[:3, :3],
                    restrict_to_roi=True,
                )
            else:
                s, t, _ = align_depth_scale_shift(
                    depth_pred=depth, ele_gt=ele_gt_np, mask_gt=mask_gt_np,
                    ground_normal_world=ground_normal, R_c2w=extr_c2w[:3, :3],
                    camera_height=camera_height, intrinsics=intr,
                )

            depth_aligned = (s * depth + t).astype(np.float32)
            H, W = depth_aligned.shape
            fx, fy = intr[0, 0], intr[1, 1]
            cx, cy = intr[0, 2], intr[1, 2]
            uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))
            zc = depth_aligned
            xc = (uu - cx) * zc / fx
            yc = (vv - cy) * zc / fy
            pts_cam = np.stack([xc, yc, zc], axis=-1).reshape(-1, 3)
            valid = (zc.reshape(-1) > 0) & np.isfinite(zc.reshape(-1))
            pts_cam = pts_cam[valid]

            ele_pred_np, mask_pred_np = pointcloud_to_height_map(
                points_cam=pts_cam,
                ground_normal_world=ground_normal,
                R_c2w=extr_c2w[:3, :3],
                camera_height=camera_height,
                y_crop=(-0.5, 2.0),
            )
            da3_pred_t = torch.from_numpy(ele_pred_np)
            da3_mask_t = torch.from_numpy(mask_pred_np & mask_gt_np)
            da3_gt_t   = torch.from_numpy(ele_gt_np)
            da3_mae, da3_rmse, _ = per_sample_errors(da3_pred_t, da3_gt_t, da3_mask_t)
        except FileNotFoundError:
            n_skip_da3 += 1
        except Exception as e:
            print(f"  [warn] da3 failed on i={i}: {e}")
            n_skip_da3 += 1

        if np.isfinite(rhf_mae) and n_rhf_cells > 0:
            rhf_mae_sum += rhf_mae
            rhf_rmse_sum += rhf_rmse
            n_rhf += 1
        else:
            n_skip_rhf += 1
        if np.isfinite(da3_mae):
            da3_mae_sum += da3_mae
            da3_rmse_sum += da3_rmse
            n_da3 += 1

        per_sample.append({
            "idx": i,
            "img_path": img_path_str,
            "rhf_mae": rhf_mae,
            "rhf_rmse": rhf_rmse,
            "da3_mae": da3_mae,
            "da3_rmse": da3_rmse,
        })

        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(test_loader)}  elapsed={time.time()-t0:.1f}s  "
                  f"rhf_avg_mae={rhf_mae_sum/max(n_rhf,1):.3f}  "
                  f"da3_avg_mae={da3_mae_sum/max(n_da3,1):.3f}")

    elapsed = time.time() - t0
    print(f"\n[done] elapsed={elapsed:.1f}s  rhf samples={n_rhf}/{len(test_loader)} (skip {n_skip_rhf})  "
          f"da3 samples={n_da3}/{len(test_loader)} (skip {n_skip_da3})")

    rhf_mae_avg  = rhf_mae_sum  / max(n_rhf, 1)
    rhf_rmse_avg = rhf_rmse_sum / max(n_rhf, 1)
    da3_mae_avg  = da3_mae_sum  / max(n_da3, 1)
    da3_rmse_avg = da3_rmse_sum / max(n_da3, 1)

    summary = {
        "rhf_baseline":   {"abs_err_cm": rhf_mae_avg, "rmse_cm": rhf_rmse_avg, "n": n_rhf},
        "da3_small":      {"abs_err_cm": da3_mae_avg, "rmse_cm": da3_rmse_avg, "n": n_da3},
        "per_sample":     per_sample,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPer-sample errors → {OUTPUT_JSON}")

    print("\n" + "=" * 72)
    print("Overall metrics (mean of per-sample MAE/RMSE)")
    print("=" * 72)
    print(f"| Model           |  MAE (cm) | RMSE (cm) |   n  |")
    print(f"|-----------------|----------:|----------:|-----:|")
    print(f"| RHF baseline    | {rhf_mae_avg:9.3f} | {rhf_rmse_avg:9.3f} | {n_rhf:4d} |")
    print(f"| DA3-SMALL (aln) | {da3_mae_avg:9.3f} | {da3_rmse_avg:9.3f} | {n_da3:4d} |")

    # ---------------- pick 10 samples ----------------
    valid = [s for s in per_sample
             if np.isfinite(s["rhf_mae"]) and np.isfinite(s["da3_mae"])]
    if not valid:
        print("\n[warn] no samples with both RHF and DA3 errors; skipping selection")
        return

    by_rhf = sorted(valid, key=lambda s: s["rhf_mae"])
    by_gap = sorted(valid, key=lambda s: (s["da3_mae"] - s["rhf_mae"]), reverse=True)
    mid = len(by_rhf) // 2

    picked, seen = [], set()
    def add(s, tag):
        if s["idx"] in seen:
            return
        seen.add(s["idx"])
        picked.append((tag, s))

    # 3 best for RHF
    for s in by_rhf[:3]:           add(s, "best_RHF")
    # 2 around the median
    add(by_rhf[mid], "median")
    add(by_rhf[mid + 1], "median")
    # 3 worst for RHF
    for s in by_rhf[-3:][::-1]:    add(s, "worst_RHF")
    # 2 largest RHF-beats-DA3 margin
    for s in by_gap[:2]:           add(s, "biggest_gain")

    print("\n" + "=" * 88)
    print("10 hand-picked samples (best / median / worst RHF + 2 biggest RHF-vs-DA3 gains)")
    print("=" * 88)
    print(f"| Tag           | idx | RHF MAE | RHF RMSE | DA3 MAE | DA3 RMSE | image |")
    print(f"|---------------|----:|--------:|---------:|--------:|---------:|-------|")
    for tag, s in picked[:10]:
        ip = (s.get("img_path") or "—")
        ip_short = ".../" + "/".join(Path(ip).parts[-3:]) if ip != "—" else "—"
        print(f"| {tag:<13} | {s['idx']:3d} | {s['rhf_mae']:7.3f} | {s['rhf_rmse']:8.3f} | "
              f"{s['da3_mae']:7.3f} | {s['da3_rmse']:8.3f} | {ip_short} |")


if __name__ == "__main__":
    main()
