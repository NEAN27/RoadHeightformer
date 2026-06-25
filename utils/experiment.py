from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import copy


def make_iterative_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


@make_iterative_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type for tensor2float")


@make_iterative_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.cpu().numpy()
    else:
        raise NotImplementedError("invalid input type for tensor2numpy")


@make_iterative_func
def check_allfloat(vars):
    assert isinstance(vars, float)


def save_scalars(logger, mode_tag, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for tag, values in scalar_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            scalar_name = '{}/{}'.format(mode_tag, tag)
                                 
            scalar_name = scalar_name + "_" + str(idx)
            print(scalar_name, value)


def save_images(logger, mode_tag, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)
    for tag, values in images_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            if len(value.shape) == 3:
                value = value[:, np.newaxis, :, :]
            value = value[:1]
            value = torch.from_numpy(value)

            image_name = '{}/{}'.format(mode_tag, tag)
            if len(values) > 1:
                image_name = image_name + "_" + str(idx)
            logger.add_image(image_name, vutils.make_grid(value, padding=0, nrow=1, normalize=True, scale_each=True),
                             global_step)


def adjust_learning_rate(optimizer, epoch, base_lr, lrepochs):
    splits = lrepochs.split(':')
    assert len(splits) == 2

                                                                
    downscale_epochs = [int(eid_str) for eid_str in splits[0].split(',')]
                                    
    downscale_rate = float(splits[1])
    print("downscale epochs: {}, downscale rate: {}".format(downscale_epochs, downscale_rate))

    lr = base_lr
    for eid in downscale_epochs:
        if epoch >= eid:
            lr /= downscale_rate
        else:
            break
    print("setting learning rate to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    def __init__(self):
        self.sum_value = 0.
        self.count = 0

    def update(self, x):
        check_allfloat(x)
        self.sum_value += x
        self.count += 1

    def mean(self):
        return self.sum_value / self.count

class AvgMteric(object):
    def __init__(self):
        self.data = None
        self.count = 0

    def update(self, x):
        self.count += 1
        if self.data is None:
            self.data = x
        else:
            for k, v in x.items():
                self.data[k] += v

    def get_metric(self):
        for k, v in self.data.items():
            self.data[k] /= self.count
        return self.data

class AverageMeterDict(object):
    def __init__(self):
        self.data = None
        self.count = 0

    def update(self, x):
        check_allfloat(x)
        self.count += 1
        if self.data is None:
            self.data = copy.deepcopy(x)
        else:
            for k1, v1 in x.items():
                if isinstance(v1, float):
                    self.data[k1] += v1
                elif isinstance(v1, tuple) or isinstance(v1, list):
                    for idx, v2 in enumerate(v1):
                        self.data[k1][idx] += v2
                else:
                    assert NotImplementedError("error input type for update AvgMeterDict")

    def mean(self):
        @make_iterative_func
        def get_mean(v):
            return v / float(self.count)

        return get_mean(self.data)


import torch.distributed as dist
def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


from collections import defaultdict
def reduce_scalar_outputs(scalar_outputs):
    world_size = get_world_size()
    if world_size < 2:
        return scalar_outputs
    with torch.no_grad():
        names = []
        scalars = []
        for k in sorted(scalar_outputs.keys()):
            if isinstance(scalar_outputs[k], (list, tuple)):
                for sub_var in scalar_outputs[k]:
                    names.append(k)
                    scalars.append(sub_var)
            else:
                names.append(k)
                scalars.append(scalar_outputs[k])

        scalars = torch.stack(scalars, dim=0)
        dist.reduce(scalars, dst=0)
        if dist.get_rank() == 0:
                                                                   
                                     
            scalars /= world_size

        reduced_scalars = defaultdict(list)
        for name, scalar in zip(names, scalars):
            reduced_scalars[name].append(scalar)

    return dict(reduced_scalars)

import torch
import matplotlib.pyplot as plt


def save_feature_map(
    feat: torch.Tensor,
    filepath: str,
    title: str = "",
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
):
    """
    Save a 2D feature map to an image file.

    Args:
        feat: Tensor of shape
              (H, W) or
              (1, H, W) or
              (B, H, W)
        filepath: output image path (e.g. "feature.png")
        title: optional title
        cmap: matplotlib colormap
        vmin/vmax: optional normalization limits
    """

    feat = feat.detach().cpu()

                          
    if feat.ndim == 3:
        feat = feat[0]

    if feat.ndim != 2:
        raise ValueError(f"Expected 2D map, got shape {feat.shape}")

    feat = feat.numpy()

    plt.figure(figsize=(6, 6))
    im = plt.imshow(feat, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")

    if title:
        plt.title(title)

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def inspect_tensor(x):
    print("max:", x.abs().max())
    print("mean:", x.mean())
    print("std:", x.std())
    print("nan:", torch.isnan(x).any())
    print("inf:", torch.isinf(x).any())