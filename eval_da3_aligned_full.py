"""Evaluate DA3-SMALL (with per-image scale+shift alignment) on the full
val_small_dataset_thesis split. Computes average metrics — no PNG/npz output.

Reuses the helpers in run_da3_inference.py:
  * align_depth_scale_shift  — solves (s, t) on GT-derived per-pixel depths
  * pointcloud_to_height_map — bins camera-frame points into the BEV grid
"""
import gzip
import os
import sys
import glob
import pickle
import time

import numpy as np
import torch
from PIL import Image

from utils.metric import Metric
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


PREPROCESSED_DIR = "/data/rhf/val_preprocessed_small_data_thesis"
MODEL_NAME = "depth-anything/DA3-SMALL"


@torch.no_grad()
def main():
    pkl_paths = sorted(glob.glob(os.path.join(PREPROCESSED_DIR, "data_item_*.pkl.gz")))
    print(f"Found {len(pkl_paths)} samples in {PREPROCESSED_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_NAME} on {device} ...")
    model = DepthAnything3.from_pretrained(MODEL_NAME).to(device=device)
    model.eval()

    # Probe one sample for the BEV grid dims (= y_range = 0.2 m, same as cardset default).
    with gzip.open(pkl_paths[0], "rb") as f:
        probe = pickle.load(f)
    nz, nx = probe["ele_gt"].shape
    print(f"BEV grid: {nz} x {nx}")
    metric = Metric(ele_range=0.2, num_grids_z=nz, distance_wise=False)

    n_done, n_skip = 0, 0
    t0 = time.time()
    for i, pkl in enumerate(pkl_paths):
        with gzip.open(pkl, "rb") as f:
            data = pickle.load(f)

        try:
            img = Image.open(data["path"]).convert("RGB").resize(RESIZE_WH)
        except Exception as e:
            n_skip += 1
            continue

        prediction = model.inference([img], process_res=RESIZE_WH[0])
        depth = prediction.depth[0].astype(np.float32)

        extr_c2w = np.asarray(data["extrinsics"], dtype=np.float32)
        intr = np.asarray(data["intrinsics"], dtype=np.float32)
        ele_gt = np.asarray(data["ele_gt"], dtype=np.float32)
        mask_gt = np.asarray(data["mask"]) > 0
        ground_normal = data["ground_normal"]
        camera_height = float(data["camera_height"])

        if not mask_gt.any():
            n_skip += 1
            continue

        # Alignment target: aggregated LiDAR point cloud restricted to the BEV ROI.
        # Falls back to the BEV-derived synthetic depths if depth_path is missing.
        pts_cam_gt = None
        dpath = data.get("depth_path")
        if dpath and os.path.exists(dpath):
            try:
                pts_cam_gt = np.load(dpath)["pts_cam"]
            except Exception:
                pts_cam_gt = None

        if pts_cam_gt is not None:
            s, t, info = align_depth_scale_shift_from_aggdepth(
                depth_pred=depth,
                pts_cam=pts_cam_gt,
                intrinsics=intr,
                ground_normal_world=ground_normal,
                R_c2w=extr_c2w[:3, :3],
                restrict_to_roi=True,
            )
        else:
            s, t, info = align_depth_scale_shift(
                depth_pred=depth,
                ele_gt=ele_gt,
                mask_gt=mask_gt,
                ground_normal_world=ground_normal,
                R_c2w=extr_c2w[:3, :3],
                camera_height=camera_height,
                intrinsics=intr,
            )
        depth_aligned = (s * depth + t).astype(np.float32)

        # Back-project aligned depth → camera frame, then bin.
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

        ele_pred, mask_pred = pointcloud_to_height_map(
            points_cam=pts_cam,
            ground_normal_world=ground_normal,
            R_c2w=extr_c2w[:3, :3],
            camera_height=camera_height,
            y_crop=(-0.5, 2.0),  # cardset default — depths are now in metres after alignment
        )

        # Feed the Metric class as it expects: tensors with a leading batch dim.
        ele_pred_t = torch.from_numpy(ele_pred).unsqueeze(0)
        ele_gt_t = torch.from_numpy(ele_gt).unsqueeze(0)
        ele_mask_t = torch.from_numpy(mask_pred & mask_gt).unsqueeze(0)
        metric.compute(ele_pred_t, ele_gt_t, ele_mask_t)

        n_done += 1
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(pkl_paths)}  elapsed={time.time()-t0:.1f}s  s_med~{s:.2f}  t~{t:+.2f}")

    metric_all, _ = metric.get_metric()
    print("\n" + "=" * 78)
    print(f"DA3-SMALL (aligned) on val_small_dataset_thesis  —  {n_done} samples ({n_skip} skipped)")
    print("=" * 78)
    print(f"  AbsErr (cm)  : {metric_all[0]:.3f}")
    print(f"  RMSE   (cm)  : {metric_all[1]:.3f}")
    print(f"  >0.5cm (frac): {metric_all[2]:.4f}  ({metric_all[2]*100:.2f}%)")
    print(f"  >0.1cm (frac): {metric_all[3]:.4f}  ({metric_all[3]*100:.2f}%)")
    print(f"  >1.0cm (frac): {metric_all[4]:.4f}  ({metric_all[4]*100:.2f}%)")
    print(f"  LE90   (cm)  : {metric_all[5]:.3f}")
    print(f"  GradErr(cm)  : {metric_all[6]:.4f}")
    print(f"  count_all    : {metric.count_all}")


if __name__ == "__main__":
    main()
