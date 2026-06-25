"""Project the camera-frame axes and the vertical-cam-frame axes on top of one
sample image to confirm which way each axis points. Saves three PNGs per idx:

    axes_idx{N}_origin_pp.png   — origin placed near the principal point (small frame at z=4 m)
    axes_idx{N}_origin_road.png — origin placed on the road, 4 m ahead
    axes_idx{N}_lidar_topdown.png — top-down (BEV) view of the LiDAR cloud in vert frame
                                    + projected ROI box and the two axis triads
"""
import gzip
import os
import pickle
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, "/home/f9ql00v/RoadHeightformer")
from run_da3_inference import _compute_R_cam2vert


PREPROCESSED_DIR = "/data/rhf/val_preprocessed_small_data_thesis"
OUT_DIR = "axes_viz"
INDICES = [131, 181]
AXIS_LEN_M = 1.0       # length of drawn axes in metres
ORIGIN_FORWARD_M = 6.0 # how far ahead of the camera to place the axis origin
DOT_R = 6              # marker radius


def project(K, p_cam):
    """Project a (3,) or (N,3) camera-frame point to (u,v) pixels.
    Returns (..., 2) array. NaN for behind-camera."""
    p = np.atleast_2d(p_cam).astype(np.float64)
    z = p[:, 2]
    u = K[0, 0] * p[:, 0] / z + K[0, 2]
    v = K[1, 1] * p[:, 1] / z + K[1, 2]
    uv = np.stack([u, v], axis=1)
    uv[z <= 0.01] = np.nan
    return uv if p_cam.ndim == 2 else uv[0]


def draw_arrow(ax, uv0, uv1, color, label, _legend_handles):
    if np.any(np.isnan(uv0)) or np.any(np.isnan(uv1)):
        return
    ax.annotate(
        "", xy=tuple(uv1), xytext=tuple(uv0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=3.5, shrinkA=0, shrinkB=0,
                        mutation_scale=22),
    )
    # Add a small clean marker at the tip — no inline text masking the arrow.
    ax.plot(uv1[0], uv1[1], "o", color=color, markersize=7,
            markeredgecolor="black", markeredgewidth=1.0, zorder=11)
    # Accumulate a legend entry instead of writing on the image.
    _legend_handles.append(plt.Line2D([0], [0], color=color, lw=3, label=label))


def make_axes_image(rgb, K, R_vert2cam, origin_cam, title, save_path):
    """Draw the cam-frame triad and the vert-frame triad starting at `origin_cam`."""
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(rgb)

    legend_handles = []

    # Camera-frame axes (in cam frame they're identity)
    cam_axes = np.eye(3, dtype=np.float64) * AXIS_LEN_M
    cam_colors = ["red", "lime", "blue"]
    cam_names  = ["+X_cam (right)", "+Y_cam (down)", "+Z_cam (forward)"]

    uv0 = project(K, origin_cam)
    for axis_dir, c, name in zip(cam_axes, cam_colors, cam_names):
        end_cam = origin_cam + axis_dir
        uv1 = project(K, end_cam)
        draw_arrow(ax, uv0, uv1, c, name, legend_handles)

    # Vertical-cam-frame axes
    vert_axes_in_vert = np.eye(3, dtype=np.float64) * AXIS_LEN_M
    vert_axes_in_cam = (R_vert2cam @ vert_axes_in_vert.T).T
    vert_colors = ["magenta", "yellow", "cyan"]
    vert_names  = ["+X_vert", "+Y_vert (= -n_road, points DOWN)", "+Z_vert (road forward)"]

    for axis_dir, c, name in zip(vert_axes_in_cam, vert_colors, vert_names):
        end_cam = origin_cam + axis_dir
        uv1 = project(K, end_cam)
        draw_arrow(ax, uv0, uv1, c, name, legend_handles)

    # Origin marker
    if not np.any(np.isnan(uv0)):
        ax.plot(uv0[0], uv0[1], "o", color="white", markersize=DOT_R,
                markeredgecolor="black", markeredgewidth=1.5, zorder=10)

    ax.legend(handles=legend_handles, loc="lower right", fontsize=10,
              framealpha=0.85, facecolor="white", edgecolor="black")
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  saved {save_path}")


def make_topdown_viz(pts_cam, R_cam2vert, save_path, title):
    """Top-down (x, z) view of the LiDAR in vertical-cam frame + axis triad + ROI box."""
    pv = pts_cam @ R_cam2vert.T
    fig, ax = plt.subplots(figsize=(11, 8))
    # Subsample for plotting speed
    n = pv.shape[0]
    if n > 80000:
        sel = np.random.choice(n, 80000, replace=False)
        pv_s = pv[sel]
    else:
        pv_s = pv
    sc = ax.scatter(pv_s[:, 0], pv_s[:, 2], c=pv_s[:, 1], cmap="plasma",
                    s=0.2, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="y_vert (m)  ‘+’ = below camera optical centre")

    # ROI box
    roi_x = (-1.5, 1.5); roi_z = (5.01, 15.0)
    ax.plot([roi_x[0], roi_x[1], roi_x[1], roi_x[0], roi_x[0]],
            [roi_z[0], roi_z[0], roi_z[1], roi_z[1], roi_z[0]],
            "g-", lw=2, label="BEV ROI (x∈[-1.5,1.5], z∈[5.01,15])")

    # Vert-frame axes at origin (camera optical centre projection)
    L = 1.5
    ax.annotate("", xy=(L, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.text(L + 0.1, 0, "+X_vert", color="red", fontsize=10)
    ax.annotate("", xy=(0, L), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="cyan", lw=2))
    ax.text(0.1, L + 0.1, "+Z_vert", color="cyan", fontsize=10)

    ax.set_xlabel("x_vert (m)  — lateral")
    ax.set_ylabel("z_vert (m)  — forward")
    ax.set_xlim(-25, 25); ax.set_ylim(-2, 40)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  saved {save_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for idx in INDICES:
        print(f"\n--- idx={idx} ---")
        pkl = os.path.join(PREPROCESSED_DIR, f"data_item_{idx:06d}.pkl.gz")
        with gzip.open(pkl, "rb") as f:
            data = pickle.load(f)

        rgb = np.array(Image.open(data["path"]).convert("RGB"))
        K = np.asarray(data["intrinsics"], dtype=np.float64)
        extr_c2w = np.asarray(data["extrinsics"], dtype=np.float64)
        gn = np.asarray(data["ground_normal"]).reshape(-1).astype(np.float64)
        ch = float(data["camera_height"])
        pts_cam = np.load(data["depth_path"])["pts_cam"].astype(np.float32)

        R_cam2vert = _compute_R_cam2vert(gn, extr_c2w[:3, :3]).astype(np.float64)
        R_vert2cam = R_cam2vert.T

        print(f"  R_cam2vert (n_vert should be [0,-1,0]):\n{R_cam2vert}")
        print(f"  Image shape: {rgb.shape}")

        # 1) Origin near the principal point, 4 m forward, on the *camera* level
        origin_pp = np.array([0.0, 0.0, ORIGIN_FORWARD_M])
        make_axes_image(
            rgb, K, R_vert2cam, origin_pp,
            title=(f"idx={idx} — axes drawn from origin at camera optical level, "
                   f"{ORIGIN_FORWARD_M:.0f} m forward\n"
                   f"red/green/blue = camera X/Y/Z   magenta/yellow/cyan = vert X/Y/Z"),
            save_path=os.path.join(OUT_DIR, f"axes_idx{idx}_origin_pp.png"),
        )

        # 2) Origin placed on the road 4 m ahead.
        # Road is at y_vert = +camera_height; in camera frame, that point is
        #   R_vert2cam @ (0, +ch, ORIGIN_FORWARD_M)
        origin_road_vert = np.array([0.0, ch, ORIGIN_FORWARD_M])
        origin_road_cam = R_vert2cam @ origin_road_vert
        make_axes_image(
            rgb, K, R_vert2cam, origin_road_cam,
            title=(f"idx={idx} — axes drawn from origin ON the ROAD, "
                   f"{ORIGIN_FORWARD_M:.0f} m forward, x=0\n"
                   f"red/green/blue = camera X/Y/Z   magenta/yellow/cyan = vert X/Y/Z"),
            save_path=os.path.join(OUT_DIR, f"axes_idx{idx}_origin_road.png"),
        )

        # 3) Top-down view of the LiDAR with the vert-frame axes
        make_topdown_viz(
            pts_cam, R_cam2vert,
            save_path=os.path.join(OUT_DIR, f"axes_idx{idx}_lidar_topdown.png"),
            title=(f"idx={idx} LiDAR in vertical-cam frame — top-down (x,z)\n"
                   f"colour = y_vert (cm-equivalent); ROI box drawn in green; "
                   f"axes at (0,0)"),
        )


if __name__ == "__main__":
    main()
