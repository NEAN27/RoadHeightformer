"""
Visualize precomputed BEV voxel-center projections on a cropped-dataset sample.

Loads one preprocessed item, applies the same road crop as the dataset
(crop_to_road=True), and overlays the projected voxel centers on the image.
Designed to produce a clean figure for the thesis.
"""

import argparse
import gzip
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


FIXED_CROP_BBOX = (613, 1199, 1607, 1703)
CROP_PATCH_ALIGN = 14
DOWN_SCALE = 4
SQUARE_SIZE = 560


def crop_to_road(img: Image.Image) -> Image.Image:
    """Replicates CARDSetDataset.crop_image when crop_to_road=True."""
    W, H = img.size
    x0, y0, x1, y1 = FIXED_CROP_BBOX
    x0 = max(0, min(int(x0), W))
    y0 = max(0, min(int(y0), H))
    x1 = max(x0, min(int(x1), W))
    y1 = max(y0, min(int(y1), H))
    cropped = img.crop((x0, y0, x1, y1))
    cw, ch = cropped.size
    align = CROP_PATCH_ALIGN
    tw = max(align, (cw // align) * align)
    th = max(align, (ch // align) * align)
    if (tw, th) != (cw, ch):
        cropped = cropped.resize((tw, th))
    return cropped


def resize_square(img: Image.Image) -> Image.Image:
    """Replicates CARDSetDataset.crop_image_square (full image -> 560x560)."""
    return img.resize((SQUARE_SIZE, SQUARE_SIZE))


def load_sample(preprocessed_dir: str, idx: int):
    path = os.path.join(preprocessed_dir, f"data_item_{idx:06d}.pkl.gz")
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    return data


def visualize(data, save_path: str, mode: str = "cropped"):
    img = Image.open(data["path"]).convert("RGB")
    if mode == "cropped":
        img = crop_to_road(img)
    elif mode == "square":
        img = resize_square(img)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'cropped' or 'square')")
    img_np = np.array(img)
    H_img, W_img = img_np.shape[:2]

    voxel_uv = np.asarray(data["voxel_uv_left"])  # feature-map coords
    u_feat, v_feat = voxel_uv[0], voxel_uv[1]
    u_img = u_feat * DOWN_SCALE
    v_img = v_feat * DOWN_SCALE

    feat_W = W_img // DOWN_SCALE
    feat_H = H_img // DOWN_SCALE
    valid = (u_feat >= 0) & (u_feat < feat_W) & (v_feat >= 0) & (v_feat < feat_H)
    valid_ratio = float(valid.mean())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img_np)
    ax.scatter(
        u_img[valid], v_img[valid],
        s=0.6, alpha=0.35, c="lime",
        label=f"in-frustum ({valid_ratio*100:.1f}%)",
    )
    if (~valid).any():
        ax.scatter(
            u_img[~valid], v_img[~valid],
            s=0.6, alpha=0.35, c="red", label="out-of-frustum",
        )
    ax.add_patch(plt.Rectangle(
        (0, 0), W_img - 1, H_img - 1,
        fill=False, ec="yellow", lw=1.0, ls="--",
    ))
    ax.set_xlim(-10, W_img + 10)
    ax.set_ylim(H_img + 10, -10)
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.set_title(
        f"Projected BEV voxel centers ({mode} sample, "
        f"{W_img}×{H_img} px, valid: {valid_ratio*100:.2f}%)"
    )
    ax.legend(markerscale=8, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {save_path}")
    print(f"image: {W_img}x{H_img} | feat: {feat_W}x{feat_H} | valid: {valid_ratio*100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default=None,
        help="Preprocessed dataset directory (auto-picked from --mode if omitted)",
    )
    parser.add_argument(
        "--mode",
        choices=["cropped", "square"],
        default="cropped",
        help="cropped = road-cropped 994x504 dataset; square = full 560x560 dataset",
    )
    parser.add_argument("--idx", type=int, default=0, help="Sample index")
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (auto-named from --mode if omitted)",
    )
    args = parser.parse_args()

    if args.dir is None:
        args.dir = (
            "/data/rhf/train_preprocessed_small_data_thesis_cropped"
            if args.mode == "cropped"
            else "/data/rhf/train_preprocessed_small_data_thesis"
        )
    if args.out is None:
        args.out = f"figures/voxel_projection_{args.mode}.png"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    data = load_sample(args.dir, args.idx)
    visualize(data, args.out, mode=args.mode)


if __name__ == "__main__":
    main()
