"""DA3 depth + height map for two automatically picked test samples (on/off road).

Selection: scan val_preprocessed_small_data_thesis, rank by std of GT height in the
BEV ROI. Lowest std = "onroad" (flat asphalt). Highest std = "offroad" (bumpy/curb).

For each picked sample: run DA3-SMALL, align (scale+shift) to GT agg_depth,
project aligned depth to road-relative height, render three panels:
  1) DA3 depth overlay (aligned to GT scale)
  2) Predicted height map over the full image
  3) Predicted height map restricted to the BEV ROI mask

Output: viz_da3_onroad_offroad.png  (2 rows x 3 cols)
"""
import gzip
import os
import pickle
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

DA3_SRC = "/home/f9ql00v/depth-anything3/Depth-Anything-3-main/src"
if DA3_SRC not in sys.path:
    sys.path.insert(0, DA3_SRC)
from depth_anything_3.api import DepthAnything3

VAL_DIR = "/data/rhf/val_preprocessed_small_data_thesis"
OUT = "viz_da3_onroad_offroad.png"
W_TARGET, H_TARGET = 560, 560
PROCESS_RES = 2254
GT_DILATE_PX = 4
H_VMAX_CLAMP = 0.5
ROI_X = (-1.5, 1.5)
ROI_Z = (5.01, 15.0)


def project_agg_depth(pts_cam, K, W, H):
    z = pts_cam[:, 2]
    in_front = z > 1e-3
    u = K[0, 0] * pts_cam[:, 0] / np.where(in_front, z, 1.0) + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / np.where(in_front, z, 1.0) + K[1, 2]
    u_i = np.round(u).astype(np.int64)
    v_i = np.round(v).astype(np.int64)
    valid = in_front & (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
    gt_depth = np.full((H, W), np.nan, dtype=np.float32)
    order = np.argsort(-z[valid])
    idxs = np.where(valid)[0][order]
    gt_depth[v_i[idxs], u_i[idxs]] = z[idxs]
    return gt_depth


def compute_R_vert2cam(n_world, R_c2w):
    n_w = n_world / np.linalg.norm(n_world)
    n_cam = R_c2w.T @ n_w
    n_cam /= np.linalg.norm(n_cam)
    if n_cam[1] > 0:
        n_cam = -n_cam
    y_vert = -n_cam
    z_cam = np.array([0, 0, 1], dtype=np.float32)
    z_vert = z_cam - np.dot(z_cam, y_vert) * y_vert
    z_vert /= np.linalg.norm(z_vert)
    x_vert = np.cross(y_vert, z_vert)
    x_vert /= np.linalg.norm(x_vert)
    return np.column_stack([x_vert, y_vert, z_vert]).astype(np.float32)


def build_roi_mask(K, R_vert2cam, W, H, road_y):
    corners_vert = np.array([
        [ROI_X[0], road_y, ROI_Z[0]],
        [ROI_X[1], road_y, ROI_Z[0]],
        [ROI_X[1], road_y, ROI_Z[1]],
        [ROI_X[0], road_y, ROI_Z[1]],
    ], dtype=np.float32)
    corners_cam = (R_vert2cam @ corners_vert.T).T
    u = K[0, 0] * corners_cam[:, 0] / corners_cam[:, 2] + K[0, 2]
    v = K[1, 1] * corners_cam[:, 1] / corners_cam[:, 2] + K[1, 2]
    poly = np.stack([u, v], axis=1)
    from matplotlib.path import Path
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    inside = Path(poly).contains_points(pts).reshape(H, W)
    return inside


def build_sky_mask(K, n_cam, W, H, margin_px=4):
    """Geometric sky mask: pixels whose viewing ray does not hit the road plane
    in front of the camera (i.e. above the horizon line). n_cam is the road
    normal in camera frame, pre-flipped so it points 'up' (n_cam[1] < 0).
    """
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rx = (uu - K[0, 2]) / K[0, 0]
    ry = (vv - K[1, 2]) / K[1, 1]
    rz = np.ones_like(rx)
    dot = n_cam[0] * rx + n_cam[1] * ry + n_cam[2] * rz
    sky = dot >= 0  # ray parallel-to or pointing-above the ground plane
    if margin_px > 0:
        from scipy.ndimage import binary_dilation
        sky = binary_dilation(sky, iterations=margin_px)
    return sky


def depth_to_height(depth, K, n_cam, camera_height):
    H, W = depth.shape
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = (uu - K[0, 2]) / K[0, 0] * depth
    y = (vv - K[1, 2]) / K[1, 1] * depth
    z = depth
    height = x * n_cam[0] + y * n_cam[1] + z * n_cam[2] + camera_height
    import pdb; pdb.set_trace()
    return height


def pick_on_off_road():
    """Rank pkls by std of GT height in BEV ROI; return (onroad_path, offroad_path)."""
    paths = sorted(glob(os.path.join(VAL_DIR, "data_item_*.pkl.gz")))
    print(f"Scanning {len(paths)} val items for on/off-road candidates...")
    stats = []
    for p in paths:
        try:
            with gzip.open(p, "rb") as f:
                d = pickle.load(f)
            ele = np.asarray(d["ele_gt"], dtype=np.float32)
            m = np.asarray(d["mask"]).astype(bool)
            if m.sum() < 500:
                continue
            vals = ele[m]
            s = float(np.std(vals))
            cov = float(m.mean())
            stats.append((p, s, cov, int(d["timestamp_us"])))
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}")
    if not stats:
        raise RuntimeError("No usable val items.")
    # Require reasonable coverage so picks aren't dominated by tiny masks.
    stats = [s for s in stats if s[2] > 0.30]
    stats.sort(key=lambda r: r[1])
    onroad = stats[0]
    offroad = stats[-1]
    print(f"  onroad  : {os.path.basename(onroad[0])}  ts={onroad[3]}  std={onroad[1]:.2f}cm  cov={onroad[2]:.2f}")
    print(f"  offroad : {os.path.basename(offroad[0])}  ts={offroad[3]}  std={offroad[1]:.2f}cm  cov={offroad[2]:.2f}")
    return onroad[0], offroad[0]


def process(pkl_path, model, device, align_to_lidar=True):
    """Run DA3 + (optional) scale+shift alignment + height projection for one pkl.

    If `align_to_lidar=False`, the model output is treated as already-metric
    (e.g. DA3NESTED-GIANT-LARGE) and the linear (a, b) fit against LiDAR depth
    is skipped, so the depth is used as-is.
    """
    with gzip.open(pkl_path, "rb") as f:
        d = pickle.load(f)
    orig_img = Image.open(d["path"]).convert("RGB")
    img_560 = orig_img.resize((W_TARGET, H_TARGET))
    img_np = np.array(img_560)

    K = d["intrinsics"].astype(np.float32).copy()
    pts_cam = np.load(d["depth_path"])["pts_cam"].astype(np.float32)
    n_w = d["ground_normal"].astype(np.float32); n_w /= np.linalg.norm(n_w)
    R_c2w = d["extrinsics"][:3, :3].astype(np.float32)
    n_cam = R_c2w.T @ n_w; n_cam /= np.linalg.norm(n_cam)
    if n_cam[1] > 0:
        n_cam = -n_cam
    camera_height = float(d["camera_height"])

    with torch.no_grad():
        pred = model.inference([orig_img], process_res=PROCESS_RES)
    da3_raw = np.asarray(pred.depth[0], dtype=np.float32)
    da3_560 = np.array(Image.fromarray(da3_raw).resize((W_TARGET, H_TARGET), Image.BILINEAR),
                       dtype=np.float32)

    gt_depth = project_agg_depth(pts_cam, K, W_TARGET, H_TARGET)
    if GT_DILATE_PX > 0:
        from scipy.ndimage import binary_dilation, grey_dilation
        mask0 = np.isfinite(gt_depth)
        mask = binary_dilation(mask0, iterations=GT_DILATE_PX)
        filled = np.where(np.isnan(gt_depth), -np.inf, gt_depth)
        gt_depth = grey_dilation(filled, size=2 * GT_DILATE_PX + 1)
        gt_depth = np.where(np.isfinite(gt_depth) & mask, gt_depth, np.nan)

    if align_to_lidar:
        pair = np.isfinite(gt_depth) & np.isfinite(da3_560)
        g = gt_depth[pair].astype(np.float64)
        p = da3_560[pair].astype(np.float64)
        A = np.stack([p, np.ones_like(p)], axis=1)
        a, b = np.linalg.lstsq(A, g, rcond=None)[0]
        da3_aligned = a * da3_560 + b
        print(f"  alignment fit:  depth = {a:.4f} * da3 + {b:+.3f}")
    else:
        a, b = 1.0, 0.0
        da3_aligned = da3_560.copy()
        print("  alignment skipped — using DA3 metric output directly (a=1, b=0)")

    pred_height = depth_to_height(da3_aligned, K, n_cam, camera_height)
    R_vert2cam = compute_R_vert2cam(n_w, R_c2w)
    roi_mask = build_roi_mask(K, R_vert2cam, W_TARGET, H_TARGET, camera_height)

    # ── GT height in image space, built directly from the aggregated LiDAR.
    # height_world(p) = camera_height + n_cam · p_cam   (same formula used by
    # depth_to_height above; positive = above the camera-derived ground plane).
    R_cam2vert = R_vert2cam.T
    gt_height_full = np.full((H_TARGET, W_TARGET), np.nan, dtype=np.float32)
    z_proj = pts_cam[:, 2]
    in_front = z_proj > 1e-3
    u_g = K[0, 0] * pts_cam[:, 0] / np.where(in_front, z_proj, 1.0) + K[0, 2]
    v_g = K[1, 1] * pts_cam[:, 1] / np.where(in_front, z_proj, 1.0) + K[1, 2]
    u_gi = np.round(u_g).astype(np.int64)
    v_gi = np.round(v_g).astype(np.int64)
    valid_g = (in_front
               & (u_gi >= 0) & (u_gi < W_TARGET)
               & (v_gi >= 0) & (v_gi < H_TARGET))
    h_pt = (n_cam[0] * pts_cam[:, 0]
            + n_cam[1] * pts_cam[:, 1]
            + n_cam[2] * pts_cam[:, 2]
            + camera_height)
    # Splat highest-z-first so nearer points overwrite (keep the surface in front).
    order = np.argsort(-z_proj[valid_g])
    idxs = np.where(valid_g)[0][order]
    gt_height_full[v_gi[idxs], u_gi[idxs]] = h_pt[idxs]

    if GT_DILATE_PX > 0:
        from scipy.ndimage import binary_dilation, grey_dilation
        gh_mask0 = np.isfinite(gt_height_full)
        gh_mask = binary_dilation(gh_mask0, iterations=GT_DILATE_PX)
        gh_fill = np.where(np.isnan(gt_height_full), -np.inf, gt_height_full)
        gh_fill = grey_dilation(gh_fill, size=2 * GT_DILATE_PX + 1)
        gt_height_full = np.where(np.isfinite(gh_fill) & gh_mask, gh_fill, np.nan)

    # Per-pixel forward distance in the vertical frame (z_vert).
    Hh, Ww = da3_aligned.shape
    vv, uu = np.meshgrid(np.arange(Hh), np.arange(Ww), indexing="ij")
    x_cam = (uu - K[0, 2]) / K[0, 0] * da3_aligned
    y_cam = (vv - K[1, 2]) / K[1, 1] * da3_aligned
    z_cam = da3_aligned
    R_cam2vert = R_vert2cam.T
    z_vert = (R_cam2vert[2, 0] * x_cam +
              R_cam2vert[2, 1] * y_cam +
              R_cam2vert[2, 2] * z_cam)

    return {
        "img": img_np,
        "ts": int(d["timestamp_us"]),
        "depth_full": da3_aligned,
        "height_full": pred_height,
        "gt_height_full": gt_height_full,
        "roi_mask": roi_mask,
        "z_vert": z_vert,
        "depth_vmin": float(np.nanmin(gt_depth)),
        "depth_vmax": float(np.nanmax(gt_depth)),
    }


def main():
    onroad_pkl, offroad_pkl = pick_on_off_road()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DA3-SMALL on {device}...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL").to(device=device).eval()

    samples = []
    for label, pkl in [("onroad", onroad_pkl), ("offroad", offroad_pkl)]:
        print(f"\n--- {label}: {os.path.basename(pkl)} ---")
        s = process(pkl, model, device)
        s["label"] = label
        samples.append(s)

    # Fixed height color range (-2 m .. +2 m) across both rows.
    h_vmin = -2.0
    h_vmax = +2.0

    fig, axes = plt.subplots(2, 5, figsize=(30, 12))
    col_titles = ["DA3 depth (aligned)",
                  "Predicted height (full image)",
                  "Predicted height (BEV ROI only)",
                  "GT height (LiDAR, image space)",
                  "Original image"]

    for r, s in enumerate(samples):
        img, roi = s["img"], s["roi_mask"]
        for c in range(5):
            ax = axes[r, c]
            if c == 0:
                im = ax.imshow(s["depth_full"], cmap="viridis",
                               vmin=s["depth_vmin"], vmax=s["depth_vmax"])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="depth [m]")
            elif c == 1:
                im = ax.imshow(s["height_full"], cmap="viridis",
                               vmin=h_vmin, vmax=h_vmax)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="height [m]")
            elif c == 2:
                roi_h = np.where(roi, s["height_full"], np.nan)
                im = ax.imshow(roi_h, cmap="viridis", vmin=h_vmin, vmax=h_vmax)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="height [m]")
            elif c == 3:
                ax.imshow(img)  # show RGB faintly under the sparse LiDAR points
                im = ax.imshow(s["gt_height_full"], cmap="viridis",
                               vmin=h_vmin, vmax=h_vmax)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="height [m]")
            else:
                ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=12)
            if c == 0:
                ax.text(-0.05, 0.5, f"{s['label']}\nts={s['ts']}",
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"\nSaved {OUT}")

    # ---- height stats at near vs far end of the ROI ----
    print("\nROI height statistics (predicted height in metres):")
    print(f"  NEAR band  : z_vert ∈ [{ROI_Z[0]:.2f}, 7.50] m  (close to the car)")
    print(f"  FAR  band  : z_vert ∈ [12.50, {ROI_Z[1]:.2f}] m  (far from the car)")
    for s in samples:
        roi = s["roi_mask"]; h = s["height_full"]; zv = s["z_vert"]
        near = roi & (zv >= ROI_Z[0]) & (zv <= 7.5) & np.isfinite(h)
        far  = roi & (zv >= 12.5)   & (zv <= ROI_Z[1]) & np.isfinite(h)

        def stats(m):
            if not m.any():
                return "no valid pixels"
            v = h[m]
            return (f"min={v.min():+.3f}  max={v.max():+.3f}  "
                    f"mean={v.mean():+.3f}  median={np.median(v):+.3f}  n={int(m.sum())}")

        print(f"\n  [{s['label']}] (ts={s['ts']})")
        print(f"     near : {stats(near)}")
        print(f"     far  : {stats(far)}")


if __name__ == "__main__":
    main()
