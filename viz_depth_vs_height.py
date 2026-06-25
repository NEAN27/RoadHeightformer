"""Visualize aggregated point cloud projected onto the image:
- depth map (z in vertical frame) with plasma colormap
- height map (-y in vertical frame, so up is positive) with plasma colormap
Goal: show that the depth range is much wider than the height range.

Usage: python viz_depth_vs_height.py [--idx N] [--split val|train] [--out PATH]
"""
import argparse
import gzip
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def compute_sky_mask(img_np):
    """HSV + flood-fill heuristic. Returns boolean mask, True where sky.

    Sky pixels are bright with low saturation (overcast) or blue-dominant
    AND connected to the top edge of the image.
    """
    import colorsys
    from scipy.ndimage import label as cc_label

    rgb = img_np.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    v = mx
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    bright_gray = (v > 0.70) & (s < 0.25)
    blue_sky = (b > r) & (b > g) & (v > 0.45)
    candidate = bright_gray | blue_sky

    H = img_np.shape[0]
    candidate[int(0.65 * H):, :] = False

    lbl, n = cc_label(candidate)
    if n == 0:
        return np.zeros(img_np.shape[:2], dtype=bool)
    top_labels = set(np.unique(lbl[0, :])) - {0}
    sky = np.isin(lbl, list(top_labels))
    return sky


def project_points(pts_cam, K, W, H):
    """Pinhole projection. pts_cam: (N,3) in camera frame (OpenCV convention)."""
    z = pts_cam[:, 2]
    in_front = z > 1e-3
    z_safe = np.where(in_front, z, 1.0)
    u = K[0, 0] * pts_cam[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z_safe + K[1, 2]
    u_i = np.round(u).astype(np.int64)
    v_i = np.round(v).astype(np.int64)
    valid = in_front & (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
    return u_i, v_i, valid


def rasterize_nearest(u_i, v_i, values, depths, valid, H, W):
    """For each pixel keep the value coming from the closest (smallest depth) point."""
    out = np.full((H, W), np.nan, dtype=np.float32)
    best_depth = np.full((H, W), np.inf, dtype=np.float32)
    idxs = np.where(valid)[0]
    order = np.argsort(-depths[idxs])
    idxs = idxs[order]
    for k in idxs:
        vv, uu = v_i[k], u_i[k]
        d = depths[k]
        if d < best_depth[vv, uu]:
            best_depth[vv, uu] = d
            out[vv, uu] = values[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--out", default="depth_vs_height.png")
    ap.add_argument("--dilate", type=int, default=1,
                    help="dilation radius in px so sparse projected points stay visible (0 = raw)")
    ap.add_argument("--interp", action="store_true", default=False,
                    help="fill empty pixels via nearest-neighbour interpolation (gives a smooth dense map)")
    ap.add_argument("--crop", default="85,155,480,400",
                    help="crop bbox in 560x560 coords: x0,y0,x1,y1 (set to '' to disable)")
    ap.add_argument("--log", action="store_true",
                    help="log-compress the colormap (LogNorm for depth, SymLogNorm for height)")
    ap.add_argument("--h-vmax", type=float, default=0.5,
                    help="height clamp: values above this saturate to plasma yellow")
    ap.add_argument("--sky-mask", action="store_true",
                    help="mask out the sky via an HSV+flood-fill heuristic and disable the rectangular crop")
    args = ap.parse_args()

    base = (
        "/data/rhf/val_preprocessed_small_data_thesis"
        if args.split == "val"
        else "/data/rhf/train_preprocessed_small_data_thesis"
    )
    pkl_path = os.path.join(base, f"data_item_{args.idx:06d}.pkl.gz")
    with gzip.open(pkl_path, "rb") as f:
        d = pickle.load(f)

    img = Image.open(d["path"]).convert("RGB")
    W_target, H_target = 560, 560
    img = img.resize((W_target, H_target))
    img_np = np.array(img)

    K = d["intrinsics"].astype(np.float32).copy()

    # Ground normal in camera frame — used to compute true road-relative height
    # (corrects for the camera pitch; naive -y would be off by z*sin(pitch)).
    n_w = d["ground_normal"].astype(np.float32)
    n_w /= np.linalg.norm(n_w)
    R_c2w = d["extrinsics"][:3, :3].astype(np.float32)
    n_cam = R_c2w.T @ n_w
    n_cam /= np.linalg.norm(n_cam)
    if n_cam[1] > 0:
        n_cam = -n_cam  # enforce "up" in camera frame
    camera_height = float(d["camera_height"])

    if args.sky_mask:
        args.crop = ""  # full frame; we mask sky instead

    if args.crop:
        x0, y0, x1, y1 = [int(v) for v in args.crop.split(",")]
        x0 = max(0, min(x0, W_target));  x1 = max(x0 + 1, min(x1, W_target))
        y0 = max(0, min(y0, H_target));  y1 = max(y0 + 1, min(y1, H_target))
        img_np = img_np[y0:y1, x0:x1]
        K[0, 2] -= x0
        K[1, 2] -= y0
        W_target, H_target = x1 - x0, y1 - y0

    pts_cam = np.load(d["depth_path"])["pts_cam"].astype(np.float32)

    depths_all = pts_cam[:, 2]
    # Signed distance from each point to the road plane (positive = above road).
    # Plane in camera frame: n_cam · X + camera_height = 0   →   h = n_cam · P + camera_height.
    heights_all = (pts_cam @ n_cam) + camera_height

    u_i, v_i, valid = project_points(pts_cam, K, W_target, H_target)

    depth_proj = depths_all[valid]
    height_proj = heights_all[valid]

    print(f"Sample: {d['path']}")
    print(f"#points total: {len(pts_cam)}  |  #projecting in image: {valid.sum()}")
    print("=" * 60)
    print("FULL aggregated point cloud (camera frame, from agg_depth)")
    print(f"  depth (z) range:  [{depths_all.min():+.3f}, {depths_all.max():+.3f}]  span={depths_all.max()-depths_all.min():.3f} m")
    print(f"  height (-y) range:[{heights_all.min():+.3f}, {heights_all.max():+.3f}]  span={heights_all.max()-heights_all.min():.3f} m")
    print("Points visible in image (in-frustum)")
    print(f"  depth range:  [{depth_proj.min():+.3f}, {depth_proj.max():+.3f}]  span={depth_proj.max()-depth_proj.min():.3f} m")
    print(f"  height range: [{height_proj.min():+.3f}, {height_proj.max():+.3f}]  span={height_proj.max()-height_proj.min():.3f} m")
    print("=" * 60)

    depth_map = rasterize_nearest(u_i, v_i, depths_all, depths_all, valid, H_target, W_target)
    height_map = rasterize_nearest(u_i, v_i, heights_all, depths_all, valid, H_target, W_target)

    if args.interp:
        from scipy.ndimage import distance_transform_edt

        v_min_visible = int(v_i[valid].min())
        sky_row = max(0, v_min_visible)

        def _fill(arr, sky):
            mask = ~np.isnan(arr)
            if not mask.any():
                return arr
            _, (ii, jj) = distance_transform_edt(~mask, return_indices=True)
            filled = arr[ii, jj]
            if sky is not None:
                filled[sky] = np.nan
            else:
                filled[:sky_row, :] = np.nan
            return filled

        sky_mask = compute_sky_mask(img_np) if args.sky_mask else None
        depth_map = _fill(depth_map, sky_mask)
        height_map = _fill(height_map, sky_mask)
    else:
        sky_mask = compute_sky_mask(img_np) if args.sky_mask else None
        if args.dilate > 0:
            from scipy.ndimage import grey_dilation
            r = args.dilate

            def _dilate(arr):
                filled = np.where(np.isnan(arr), -np.inf, arr)
                dil = grey_dilation(filled, size=(2 * r + 1, 2 * r + 1))
                dil = np.where(np.isfinite(dil), dil, np.nan)
                return dil

            depth_map = _dilate(depth_map)
            height_map = _dilate(height_map)
        if sky_mask is not None:
            depth_map[sky_mask] = np.nan
            height_map[sky_mask] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    axes[0].imshow(img_np)
    axes[0].axis("off")

    from matplotlib.colors import LogNorm, SymLogNorm, Normalize

    hfin = height_map[np.isfinite(height_map)]
    h_vmin = float(hfin.min()) if hfin.size else -1.0
    h_vmax = float(args.h_vmax)

    if args.log:
        dpos = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
        d_vmin = max(float(dpos.min()), 1e-3) if dpos.size else 1.0
        d_vmax = float(np.nanmax(depth_map))
        depth_norm = LogNorm(vmin=d_vmin, vmax=d_vmax)
        linthresh = max(min(abs(h_vmin), abs(h_vmax)) * 0.3, 1e-2)
        height_norm = SymLogNorm(linthresh=linthresh, vmin=h_vmin, vmax=h_vmax, base=10)
    else:
        depth_norm = Normalize(vmin=np.nanmin(depth_map), vmax=np.nanmax(depth_map))
        height_norm = Normalize(vmin=h_vmin, vmax=h_vmax)

    axes[1].imshow(img_np)
    im1 = axes[1].imshow(depth_map, cmap="plasma", alpha=0.85, norm=depth_norm)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02,
                 label="z [m]" + (" (log)" if args.log else ""))

    axes[2].imshow(img_np)
    im2 = axes[2].imshow(height_map, cmap="plasma", alpha=0.85, norm=height_norm)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02,
                 label="height [m]" + (" (symlog)" if args.log else ""))

    plt.tight_layout()
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
