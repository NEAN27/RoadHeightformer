"""Forward-pass one image (square-crop 560x560, no crop_to_road) through three
encoders and dump a PCA-RGB visualization of their feature maps:

    DINOv2_fb (facebook DINOv2 ViT-S/14 + Patch2Feature)
    EfficientNet (RoadBEV-style efficientnet_feature)
    DA3 Dino     (DepthAnything3 internal DINOv2 encoder + Patch2Feature)

For the two DINO-based encoders the visualised tensor is the output of the
Patch2Feature upsampler (i.e. the [B, pred_dim, H/4, W/4] feature map fed to
the ele_head), as requested. For EfficientNet the encoder output itself is
visualised (no upsampler in that branch).

Outputs are written to figures/encoder_features/<sample_tag>/:
    input.png
    DINOv2_fb_features.png
    EfficientNet_features.png
    DA3_dino_features.png
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

from utils.config import create_parser, load_config, update_args_with_config
from cardset.dataset import CARDSetDataset
from models.model_dinov2_fb import Elevation as ElevationDinoV2FB
from models.model import Elevation as ElevationDA3


CARDSET_ROOT = "/data/T7/cariad dataset"
DEFAULT_REL = "germany_batch2/Ludwig6/img/cam_1/cam_1_9401807.jpg"

DINOV2_CKPT = "/data/rhf/checkpoints/RHF_compositeloss_dinov2_rhf_baseline/final_RHF_compositeloss_dinov2_rhf_baseline_epoch30_007860.pt"
DINOV2_CFG = "configs/config_freeze_baseline.yaml"
ROADBEV_CKPT = "/data/rhf/checkpoints/RoadBEV_cardset_epoch30/final_RoadBEV_cardset_epoch30_007860.pt"
ROADBEV_CFG = "configs/config_roadbev_cardset.yaml"
DA3_CFG = "configs/config_da3.yaml"


def args_from_config(yaml_path):
    parser = create_parser()
    args = parser.parse_args([])
    config = load_config(yaml_path)
    args = update_args_with_config(args, config)
    args.down_scale = 4
    return args


def make_dataset(rel_path, args):
    """Build a 1-item CARDSetDataset around the provided relative image path."""
    tmp_split = "/tmp/_viz_encoder_features.txt"
    with open(tmp_split, "w") as f:
        f.write(rel_path + "\n")
    return CARDSetDataset(
        root_dir=CARDSET_ROOT,
        split_file=tmp_split,
        mode="test",
        down_scale=args.down_scale,
        preprocessed_data=False,
        augmentation=False,
        clamp_gt=False,
        crop_to_road=False,
    )


def pca_to_rgb(features_chw):
    """[C, H, W] tensor -> [H, W, 3] uint8 PCA-RGB visualization (per-channel norm)."""
    f = features_chw.detach().cpu().numpy()
    C, H, W = f.shape
    flat = f.transpose(1, 2, 0).reshape(-1, C)
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(flat).reshape(H, W, 3)
    mn = rgb.min(axis=(0, 1), keepdims=True)
    mx = rgb.max(axis=(0, 1), keepdims=True)
    rgb = (rgb - mn) / (mx - mn + 1e-8)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def save_panel(path, rgb, title=None):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_input(path, img_chw):
    img = img_chw.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = (img.clip(0, 1) * 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


@torch.no_grad()
def run_dinov2_fb(sample, args, ckpt_path):
    ds_args = sample["ds"]
    img, ele_gt, ele_mask, proj_idx, _ = sample["batch"]
    num_grids = ds_args["num_grids"]; ele_range = ds_args["ele_range"]

    model = ElevationDinoV2FB(
        stereo=False, num_grids=num_grids, ele_range=ele_range,
        cla_res=args.cla_res, regression=args.regression, backbone=args.backbone,
        normalize=args.normalize, pred_dim=args.pred_head_dim,
        train_encoder=args.train_encoder,
        dinov2_layers=tuple(args.dinov2_layers),
        upsampler_kind=args.upsampler_kind,
    ).cuda().eval()
    ckpt = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(ckpt["model"], strict=True)

    img = img.cuda(); proj_idx = proj_idx.cuda()
    _ = model(img, proj_idx)
    return model._last_features[0]   # [C, H/4, W/4]


@torch.no_grad()
def run_efficientnet(sample, args, ckpt_path):
    ds_args = sample["ds"]
    img, ele_gt, ele_mask, proj_idx, _ = sample["batch"]
    num_grids = ds_args["num_grids"]; ele_range = ds_args["ele_range"]

    model = ElevationDA3(
        stereo=False, num_grids=num_grids, ele_range=ele_range,
        cla_res=args.cla_res, regression=args.regression, backbone=args.backbone,
        normalize=args.normalize, pred_dim=args.pred_head_dim,
        train_encoder=args.train_encoder,
    ).cuda().eval()
    ckpt = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(ckpt["model"], strict=True)

    img = img.cuda(); proj_idx = proj_idx.cuda()
    _ = model(img, proj_idx)
    return model._last_features[0]


@torch.no_grad()
def run_da3(sample, args):
    ds_args = sample["ds"]
    img, ele_gt, ele_mask, proj_idx, _ = sample["batch"]
    num_grids = ds_args["num_grids"]; ele_range = ds_args["ele_range"]

    # No RHF_DA3 ckpt is available — the DepthAnything3 encoder is loaded
    # pretrained inside the constructor; the patch2feature upsampler is
    # randomly initialized. The PCA visualisation therefore reflects the
    # DINOv2 encoder structure projected through a random linear map, which
    # still preserves the spatial layout but should be read with that caveat.
    model = ElevationDA3(
        stereo=False, num_grids=num_grids, ele_range=ele_range,
        cla_res=args.cla_res, regression=args.regression, backbone=args.backbone,
        normalize=args.normalize, pred_dim=args.pred_head_dim,
        train_encoder=args.train_encoder,
    ).cuda().eval()

    img = img.cuda(); proj_idx = proj_idx.cuda()
    _ = model(img, proj_idx)
    return model._last_features[0]


def load_one_sample(rel_path, cfg_path):
    args = args_from_config(cfg_path)
    ds = make_dataset(rel_path, args)
    sample = ds[0]
    # CARDSetDataset 'test' returns (img, ele_gt, ele_mask, proj_index, time)
    batch = tuple(t.unsqueeze(0) if torch.is_tensor(t) else t for t in sample)
    ds_args = {
        "ele_range": ds.y_range,
        "num_grids": [ds.num_grids_x, ds.num_grids_y, ds.num_grids_z],
    }
    return {"batch": batch, "ds": ds_args}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel_path", default=DEFAULT_REL,
                    help="Image path relative to the cariad dataset root")
    ap.add_argument("--out_dir", default=None,
                    help="Output folder; defaults to figures/encoder_features/<basename>")
    args_cli = ap.parse_args()

    base = os.path.splitext(os.path.basename(args_cli.rel_path))[0]
    out_dir = args_cli.out_dir or os.path.join("figures", "encoder_features", base)
    os.makedirs(out_dir, exist_ok=True)

    # The dataset enforces preprocessing per-config; we run each backbone via
    # its own config to keep image normalisation / proj_index consistent with
    # what each model was trained with.
    print(f"== Sample: {args_cli.rel_path}")
    print(f"== Output: {out_dir}")

    # 1) DINOv2_fb
    print("\n[DINOv2_fb] forward pass ...")
    sample = load_one_sample(args_cli.rel_path, DINOV2_CFG)
    args = args_from_config(DINOV2_CFG)
    feat = run_dinov2_fb(sample, args, DINOV2_CKPT)
    print(f"  feature map shape: {tuple(feat.shape)}")
    save_input(os.path.join(out_dir, "input.png"), sample["batch"][0][0])
    save_panel(os.path.join(out_dir, "DINOv2_fb_features.png"),
               pca_to_rgb(feat),
               "DINOv2_fb (facebook ViT-S/14) + Patch2Feature")

    # 2) EfficientNet (via the trained RoadBEV checkpoint)
    print("\n[EfficientNet] forward pass ...")
    sample = load_one_sample(args_cli.rel_path, ROADBEV_CFG)
    args = args_from_config(ROADBEV_CFG)
    feat = run_efficientnet(sample, args, ROADBEV_CKPT)
    print(f"  feature map shape: {tuple(feat.shape)}")
    save_panel(os.path.join(out_dir, "EfficientNet_features.png"),
               pca_to_rgb(feat),
               "EfficientNet feature map")

    # 3) DA3 Dino (no trained head -> random patch2feature; encoder is pretrained)
    print("\n[DA3 Dino] forward pass ...")
    sample = load_one_sample(args_cli.rel_path, DA3_CFG)
    args = args_from_config(DA3_CFG)
    feat = run_da3(sample, args)
    print(f"  feature map shape: {tuple(feat.shape)}")
    save_panel(os.path.join(out_dir, "DA3_dino_features.png"),
               pca_to_rgb(feat),
               "DA3 (DepthAnything3) Dino encoder + Patch2Feature")

    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
