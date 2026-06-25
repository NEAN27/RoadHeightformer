"""Re-evaluate every available `final_*.pt` checkpoint trained on CARDSetSmall
on the same val split, and report all available metrics in a markdown table.

Skips RSRD-trained ckpts (different dataset).
Handles three backbones: DINOv2_fb, DA3-SMALL/DA3, EfficientNet.
"""
import json
import os
import sys
import time
import traceback

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/f9ql00v/RoadHeightformer")

from utils.config import create_parser, load_config, update_args_with_config
from utils.metric import Metric
from cardset.dataset import CARDSetDataset
from models.model_dinov2_fb import Elevation as ElevationDinoV2FB
from models.model import Elevation as ElevationDA3


CARDSET_ROOT = "/data/T7/cariad dataset"
VAL_SPLIT    = "/data/rhf/val_small_dataset_thesis.txt"
PREP_DIR     = "/data/rhf/val_preprocessed_small_data_thesis"
CONFIG_DIR   = "configs"

# (label, ckpt_path, config_yaml)
RUNS = [
    # --- RHF baseline (freeze) ---
    ("RHF_baseline",
     "/data/rhf/checkpoints/RHF_compositeloss_dinov2_rhf_baseline/final_RHF_compositeloss_dinov2_rhf_baseline_epoch30_007860.pt",
     "config_freeze_baseline.yaml"),

    # --- RHF ablations (DINOv2_fb backbone, various design knobs) ---
    ("RHF_clamp_gt",
     "/data/rhf/checkpoints/RHF_clamp_gt/final_RHF_clamp_gt_epoch30_007860.pt",
     "config_clamp_gt.yaml"),
    ("RHF_composite_l1_only",
     "/data/rhf/checkpoints/RHF_composite_l1_only/final_RHF_composite_l1_only_epoch30_007860.pt",
     "config_composite_l1_only.yaml"),
    ("RHF_composite_mse_only",
     "/data/rhf/checkpoints/RHF_composite_mse_only/final_RHF_composite_mse_only_epoch30_007860.pt",
     "config_composite_mse_only.yaml"),
    ("RHF_crop_to_road",
     "/data/rhf/checkpoints/RHF_crop_to_road/final_RHF_crop_to_road_epoch30_007860.pt",
     "config_crop_to_road.yaml"),
    ("RHF_dinov2_dino_upsampler",
     "/data/rhf/checkpoints/RHF_dinov2_dino_upsampler/final_RHF_dinov2_upsampler_dino_epoch30_007860.pt",
     "config_dinov2_upsampler.yaml"),
    ("RHF_dinov2_layers_11",
     "/data/rhf/checkpoints/RHF_dinov2_layers_11_11_11_11/final_RHF_dinov2_layers_11_11_11_11_epoch30_007860.pt",
     "config_dinov2_intermediate_11s.yaml"),
    ("RHF_dinov2_trainable",
     "/data/rhf/checkpoints/RHF_dinov2_trainable/final_RHF_dinov2_trainable_epoch30_007860.pt",
     "config_dinov2_trainable.yaml"),

    # --- Backbone swap ablations ---
    ("RHF_dinov2fb_classification",
     "/data/rhf/checkpoints/final_RHF_dinov2fb_classification/final_RHF_dinov2fb_classification_epoch30_007860.pt",
     "config_dinov2fb_classification.yaml"),
    ("RHF_efficientnet",
     "/data/rhf/checkpoints/final_RHF_efficientnet/final_RHF_efficientnet_epoch30_007860.pt",
     "config_efficientnet.yaml"),
    ("RHF_DepthAnything3",
     "/data/rhf/checkpoints/20260507123931/final_RHF_DepthAnything3_epoch30_007860.pt",
     "config_da3.yaml"),

    # --- External baseline (final RoadBEV trained on CARDSet) ---
    ("RoadBEV",
     "/data/rhf/checkpoints/RoadBEV_cardset_epoch30/final_RoadBEV_cardset_epoch30_007860.pt",
     "config_roadbev_cardset.yaml"),
]


def args_from_config(yaml_path):
    parser = create_parser()
    a = parser.parse_args([])
    a = update_args_with_config(a, load_config(yaml_path))
    a.down_scale = 4
    return a


def build_model(args, ds):
    num_grids = [ds.num_grids_x, ds.num_grids_y, ds.num_grids_z]
    ele_range = ds.y_range
    if "DINOv2_fb" in args.backbone:
        m = ElevationDinoV2FB(
            stereo=False, num_grids=num_grids, ele_range=ele_range,
            cla_res=args.cla_res, regression=args.regression,
            backbone=args.backbone, normalize=args.normalize,
            pred_dim=args.pred_head_dim, train_encoder=args.train_encoder,
            dinov2_layers=tuple(args.dinov2_layers),
            upsampler_kind=args.upsampler_kind,
        ).cuda()
    else:
        m = ElevationDA3(
            stereo=False, num_grids=num_grids, ele_range=ele_range,
            cla_res=args.cla_res, regression=args.regression,
            backbone=args.backbone, normalize=args.normalize,
            pred_dim=args.pred_head_dim, train_encoder=args.train_encoder,
        ).cuda()
    return m


@torch.no_grad()
def run_one(label, ckpt, args):
    print(f"\n{'='*78}")
    print(f"[ {label} ]")
    print(f"  ckpt    : {ckpt}")
    print(f"  backbone: {args.backbone}  regression={args.regression}  "
          f"clamp_gt={args.clamp_gt}  crop_to_road={args.crop_to_road}")
    test_set = CARDSetDataset(
        root_dir=CARDSET_ROOT, split_file=VAL_SPLIT, mode="test",
        down_scale=args.down_scale, preprocessed_data=args.preprocessed,
        augmentation=False, clamp_gt=args.clamp_gt,
        crop_to_road=args.crop_to_road,
    )
    test_set.preprocessed_dir = PREP_DIR
    loader = DataLoader(test_set, 1, shuffle=False, num_workers=4,
                        drop_last=False, pin_memory=False)
    print(f"  N test  : {len(test_set)}")

    model = build_model(args, test_set)
    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd, strict=True)
    model.eval()

    ele_range = test_set.y_range
    metric = Metric(ele_range, test_set.num_grids_z, distance_wise=False)

    t0 = time.time()
    n_done = 0
    for i, sample in enumerate(loader):
        try:
            imgs_left, ele_gt, ele_mask, uv, _ = sample
        except Exception:
            continue
        imgs_left = imgs_left.cuda(non_blocking=True)
        ele_gt = ele_gt.cuda(non_blocking=True)
        ele_mask = ele_mask.cuda(non_blocking=True)
        uv = uv.cuda(non_blocking=True)
        pred = model(imgs_left, uv)
        metric.compute(pred, ele_gt, ele_mask)
        n_done += 1
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(loader)}  elapsed={time.time()-t0:.0f}s")
    [metric_all, _] = metric.get_metric()
    print(f"  done    : {n_done}/{len(loader)}  elapsed={time.time()-t0:.1f}s")
    return {
        "abs_err":  float(metric_all[0]),
        "rmse":     float(metric_all[1]),
        "ratio_05": float(metric_all[2]),
        "ratio_01": float(metric_all[3]),
        "ratio_10": float(metric_all[4]),
        "le90":     float(metric_all[5]),
        "grad_err": float(metric_all[6]),
        "n":        int(metric.count_all),
    }


def main():
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    out = {}
    for label, ckpt, cfg_name in RUNS:
        cfg = os.path.join(CONFIG_DIR, cfg_name)
        if not os.path.isfile(ckpt):
            print(f"[skip] {label}: ckpt missing — {ckpt}")
            out[label] = {"error": f"ckpt missing: {ckpt}"}; continue
        if not os.path.isfile(cfg):
            print(f"[skip] {label}: config missing — {cfg}")
            out[label] = {"error": f"cfg missing: {cfg}"}; continue
        try:
            a = args_from_config(cfg)
            out[label] = run_one(label, ckpt, a)
        except Exception as e:
            print(f"[err] {label}: {e}")
            traceback.print_exc()
            out[label] = {"error": str(e)}
        torch.cuda.empty_cache()

    raw_json = "/home/f9ql00v/RoadHeightformer/eval_all_finals_results.json"
    with open(raw_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[json] -> {raw_json}")

    print("\n" + "=" * 110)
    print(f"Final ckpt evaluation on CARDSetSmall val split ({VAL_SPLIT})")
    print("=" * 110)
    print(f"| Run | AbsErr (cm) | RMSE (cm) | LE90 (cm) | GradErr (cm) | "
          f">0.1cm (%) | >0.5cm (%) | >1.0cm (%) | n |")
    print(f"|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, _, _ in RUNS:
        r = out.get(label, {})
        if "error" in r:
            print(f"| {label} | — | — | — | — | — | — | — | ERROR: {r['error'][:55]} |")
        else:
            print(f"| {label} | {r['abs_err']:.3f} | {r['rmse']:.3f} | {r['le90']:.3f} | "
                  f"{r['grad_err']:.4f} | {r['ratio_01']*100:.2f} | "
                  f"{r['ratio_05']*100:.2f} | {r['ratio_10']*100:.2f} | {r['n']} |")


if __name__ == "__main__":
    main()
