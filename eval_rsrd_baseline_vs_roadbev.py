"""Compare RHF_baseline_rsrd vs. RoadBEV_rsrd on the RSRD-dense test split.

The bundled RSRD class hard-codes filenames/train/ for training=False as a
debug shortcut (utils/dataset.py:41). We subclass it to point at the real
test split (filenames/test/, preprocessed/test/).
"""
import json
import os
import time
import traceback

import torch
from torch.utils.data import DataLoader

from utils.config import create_parser, load_config, update_args_with_config
from utils.dataset import RSRD
from utils.metric import Metric
from models.model_dinov2_fb import Elevation as ElevationDinoV2FB
from models.model import Elevation as ElevationDA3


class RSRDTest(RSRD):
    """Override the debug-only training=False branch to use the real test split."""
    def __init__(self, stereo=False, down_scale=2, backbone=None):
        # bypass parent's overridden filenames/train/ behavior
        super().__init__(training=True, stereo=stereo, down_scale=down_scale, backbone=backbone)
        self.training = False
        self.load_dataset_names('./filenames/test/')
        self.preprocessed_path = os.path.join('./preprocessed/', 'test')


RUNS = [
    {
        "label": "RHF_baseline_rsrd",
        "config": "configs/config_baseline_rsrd.yaml",
        "ckpt": "/data/rhf/checkpoints/RHF_baseline_rsrd/final_RHF_baseline_rsrd_epoch30_004530.pt",
    },
    {
        "label": "RoadBEV_rsrd",
        "config": "configs/config_roadbev_rsrd.yaml",
        "ckpt": "/data/rhf/checkpoints/RoadBEV_rsrd/final_RoadBEV_rsrd_epoch30_004530.pt",
    },
]


def args_from_config(yaml_path):
    parser = create_parser()
    args = parser.parse_args([])
    config = load_config(yaml_path)
    args = update_args_with_config(args, config)
    args.down_scale = 4  # mono path, matches test.py
    return args


@torch.no_grad()
def run_one(label, ckpt_path, args):
    print(f"\n{'='*72}\n[ {label} ]\n  ckpt={ckpt_path}\n{'='*72}")
    print(f"  backbone={args.backbone}  regression={args.regression}")

    test_set = RSRDTest(stereo=False, down_scale=args.down_scale, backbone=args.backbone)
    test_loader = DataLoader(test_set, 1, shuffle=False, num_workers=4, drop_last=False, pin_memory=False)
    print(f"  test set size: {len(test_set)}")

    ele_range = test_set.y_range
    num_grids = [test_set.num_grids_x, test_set.num_grids_y, test_set.num_grids_z]

    if 'DINOv2_fb' in args.backbone:
        model = ElevationDinoV2FB(
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
        ).cuda()
    else:
        model = ElevationDA3(
            stereo=False,
            num_grids=num_grids,
            ele_range=ele_range,
            cla_res=args.cla_res,
            regression=args.regression,
            backbone=args.backbone,
            normalize=args.normalize,
            pred_dim=args.pred_head_dim,
            train_encoder=args.train_encoder,
        ).cuda()
    model.eval()

    ckpt = torch.load(ckpt_path, map_location='cuda')
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)

    metric = Metric(ele_range, test_set.num_grids_z, distance_wise=False)

    t0 = time.time()
    n_done, n_skip = 0, 0
    for i, sample in enumerate(test_loader):
        try:
            imgs_left, ele_gt, ele_mask, proj_index_left, _ = sample
        except Exception:
            n_skip += 1
            continue
        imgs_left = imgs_left.cuda()
        ele_gt = ele_gt.cuda()
        ele_mask = ele_mask.cuda()
        proj_index_left = proj_index_left.cuda()

        pred = model(imgs_left, proj_index_left)
        metric.compute(pred, ele_gt, ele_mask)
        n_done += 1
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(test_loader)}  elapsed={time.time()-t0:.1f}s")

    [metric_all, _] = metric.get_metric()
    print(f"  done: {n_done} samples ({n_skip} skipped), elapsed={time.time()-t0:.1f}s, count_all={metric.count_all}")
    return {
        "abs_err":   float(metric_all[0]),
        "rmse":      float(metric_all[1]),
        "ratio_05":  float(metric_all[2]),
        "ratio_01":  float(metric_all[3]),
        "ratio_10":  float(metric_all[4]),
        "le90":      float(metric_all[5]),
        "grad_err":  float(metric_all[6]),
        "n_samples": int(metric.count_all),
    }


def main():
    torch.backends.cudnn.benchmark = True

    results = {}
    for run in RUNS:
        label, cfg, ckpt = run["label"], run["config"], run["ckpt"]
        if not os.path.isfile(ckpt):
            results[label] = {"error": f"ckpt missing: {ckpt}"}
            continue
        if not os.path.isfile(cfg):
            results[label] = {"error": f"config missing: {cfg}"}
            continue
        try:
            args = args_from_config(cfg)
            results[label] = run_one(label, ckpt, args)
        except Exception as e:
            print(f"[error] {label}: {e}")
            traceback.print_exc()
            results[label] = {"error": str(e)}
        torch.cuda.empty_cache()

    out_json = "eval_rsrd_baseline_vs_roadbev_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results -> {out_json}")

    print("\n" + "="*78)
    print("Comparison on RSRD-dense test split (filenames/test/)")
    print("="*78)
    print("| Run | abs_err (cm) | rmse (cm) | le90 (cm) | grad_err (cm) | >0.5cm (%) | n |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for run in RUNS:
        label = run["label"]
        r = results.get(label, {})
        if "error" in r:
            print(f"| {label} | — | — | — | — | — | ERROR: {r['error'][:60]} |")
        else:
            print(f"| {label} | {r['abs_err']:.3f} | {r['rmse']:.3f} | {r['le90']:.3f} | {r['grad_err']:.4f} | {r['ratio_05']*100:.2f} | {r['n_samples']} |")


if __name__ == "__main__":
    main()
