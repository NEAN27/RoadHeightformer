import argparse
import shutil
import torch.nn as nn

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from utils.dataset import RSRD
from torch.cuda.amp import GradScaler
from models.loss import MyLoss
from torch.utils.data import DataLoader
from models.model import Elevation as ElevationDA3
from models.model_dinov2_fb import Elevation as ElevationDinoV2FB
import pickle
import os
from utils.metric import Metric
from utils.experiment import *
import numpy as np
from cardset.dataset import CARDSetDataset, CARDSetDatasetV2Smalldataset

def unnormalize(ele_pred, h_min, h_max):
    height = ele_pred[:, 0:1]                    
    height = height * ((h_max - h_min) / 2) + ((h_max + h_min) / 2)
    ele_pred = torch.cat([height, ele_pred[:, 1:2]], dim=1)

    return ele_pred

@make_nograd_func
def test_sample(test_loader):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = torch.zeros(len(test_loader))
    model.eval()
    for i, sample in enumerate(test_loader):
        if args.stereo:
            print("PPPPPPPPPPPPPPPPPPP",i)
            (imgs_left, imgs_right, ele_gt, ele_mask, proj_index_left, proj_index_right, cur_time) = sample
            imgs_right, proj_index_right = imgs_right.cuda(), proj_index_right.cuda()
        else:
            (imgs_left, ele_gt, ele_mask, proj_index_left, cur_time) = sample
        imgs_left, ele_gt, ele_mask, proj_index_left = imgs_left.cuda(), ele_gt.cuda(), ele_mask.cuda(), proj_index_left.cuda()

        starter.record()
        if args.stereo:
            pred = model(imgs_left, proj_index_left, imgs_right, proj_index_right)
        else:
            pred = model(imgs_left, proj_index_left)

        
        print("predictions",pred.shape)
        print("Ground truth",ele_gt.shape)

        vmin = torch.min(ele_gt[ele_mask>0]).item()
        vmax = torch.max(ele_gt[ele_mask>0]).item()

        vmin = max(vmin, -ele_range*100)
        vmax = min(vmax, ele_range*100)

        if args.normalize:
            h_min = - ele_range * 100
            h_max = ele_range * 100                        
            pred = unnormalize(pred, h_min, h_max)


        if args.regression:
            None                                   
           
        if i % 10 == 0:
            CARDSetDatasetV2Smalldataset.visualize_height_map_and_mask(pred.squeeze(), ele_mask.squeeze(), colormap='plasma', save_path='Testimage/' + str(cur_time.item()) + '_pred', vmin= vmin, vmax=vmax)
            CARDSetDatasetV2Smalldataset.visualize_height_map_and_mask(ele_gt.squeeze(), ele_mask.squeeze(), colormap='plasma', save_path='Testimage/' + str(cur_time.item()) + '_gt', vmin= vmin, vmax=vmax )
        
        ender.record()
        torch.cuda.synchronize()
        times[i] = starter.elapsed_time(ender)

        print(ele_gt.shape)
        metric.compute(pred, ele_gt, ele_mask)
                                                                    
                                                      
    mean_time = times.mean().item()
    print("Inference time: {:.2f}ms, FPS: {:.2f} ".format(mean_time, 1000 / mean_time))
    print(metric.count_all)
    metric_values = metric.get_metric()
    return metric_values

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Elevation')
    parser.add_argument('--stereo', action='store_true', help='if yes, use RoadBEV-stereo; otherwise, RoadBEV-mono')
    parser.add_argument('--cla_res', type=float, default=0.5, help='class resolution for elevation classification')
    parser.add_argument('--loadckpt', default='./checkpoints/20240407064559/checkpoint_epoch50_007500.ckpt', help='load the weights from a specific checkpoint')
    parser.add_argument('--seed', type=int, default=837, metavar='S', help='random seed')
    parser.add_argument('--regression', action='store_true', help='regression or classification')
    parser.add_argument('--backbone',default='efficientnet', help='Use DepthAnything3 backbone or EfficientNet')
    parser.add_argument('--normalize', action='store_true', help='if set, normalize the height values to [-1, 1] for regression')
    parser.add_argument('--dataset', help='dataset to use: add it to wandb runs')
    parser.add_argument('--pred_head_dim', type=int, default=128, help='define the bottleneck between the transformer encoder and the CNN prediction head')
    parser.add_argument('--preprocessed', action='store_true', help='if yes, the dataloader will load preprocessed data')
    parser.add_argument('--load_pt', default=None, help='load weights, optimizer, start_idx to resume run')
    parser.add_argument('--dino', default="small", help='ViT encoder size')
    parser.add_argument('--clamp_gt', action='store_true', help='if set, clamp GT elevation values to [-y_range*100, y_range*100] cm in the dataloader (in addition to the existing ROI mask filtering)')
    parser.add_argument('--crop_to_road', action='store_true', help='if set, dataloader crops each image to the projected voxel ROI (+10% padding), resizes back to 560x560, and adjusts intrinsic / voxel_uv accordingly. Preprocessed cache must be regenerated when toggling this flag.')

                                
    args = parser.parse_args()
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs('./bev_pred/', exist_ok=True)
    
    if args.stereo:
        args.down_scale = 2
        print('Testing RoadBEV-stereo!')
    else:
        args.down_scale = 4
        print('Testing RoadBEV-mono!')

                         
    if 'CARDSetV2Small' == args.dataset:
        test_set = CARDSetDatasetV2Smalldataset(root_dir='CARDSet/CARD_nice', mode='test', down_scale=args.down_scale, clamp_gt=args.clamp_gt, crop_to_road=args.crop_to_road)

    elif 'CARDSet_y04_g40_square' == args.dataset:
        print("Preprocessed dataset y0.4 g40 square")
        test_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', split_file='/data/rhf/val_dataset_y0.4_g40_square.txt', mode='test', down_scale=args.down_scale, preprocessed_data=args.preprocessed, augmentation=False, clamp_gt=args.clamp_gt, crop_to_road=args.crop_to_road)
        test_set.preprocessed_dir = '/data/rhf/val_preprocessed_data_y0.4_g40_square'

    elif 'CARDSetSmall' == args.dataset:
        print("Small preprocessed (thesis) dataset")
        test_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', split_file='/data/rhf/val_small_dataset_thesis.txt', mode='test', down_scale=args.down_scale, preprocessed_data=args.preprocessed, augmentation=False, clamp_gt=args.clamp_gt, crop_to_road=args.crop_to_road)

    elif "CARDSet" == args.dataset:
        test_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', split_file='/data/T7/cariad dataset/val_all_data_clean_NN_RHF.txt', mode='test', down_scale=args.down_scale, clamp_gt=args.clamp_gt, crop_to_road=args.crop_to_road)

                                                                                                                                                                               
    elif 'RSRD' == args.dataset:
        test_set = RSRD(training=False, stereo=args.stereo, down_scale=args.down_scale, backbone=args.backbone)

    else:
        print("unknown dataset")
        exit(0)

                                                                                    
    test_loader = DataLoader(test_set, 1, shuffle=False, num_workers=4, drop_last=False, pin_memory=False)
    print('test set:', len(test_set))
    log_dir = "testing_files"
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(os.path.join(log_dir, 'log.txt'), 'a')
    
           
    ele_range = test_set.y_range
    voxel_ele_res = test_set.grid_res[1]
    num_grids = [test_set.num_grids_x, test_set.num_grids_y, test_set.num_grids_z]


    Elevation = ElevationDinoV2FB if 'DINOv2_fb' in args.backbone else ElevationDA3
    model = Elevation(args.stereo, num_grids, ele_range, args.cla_res, args.regression, args.backbone, args.normalize, args.pred_head_dim).cuda()
    print(model)
    print('num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    metric = Metric(ele_range, test_set.num_grids_z, distance_wise=False)

                               
    print("loading model {}".format(args.loadckpt))
    checkpoint = torch.load(args.load_pt)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict, strict=True)

    [metric_all, metric_depthwise] = test_sample(test_loader)
    info = ('test:    abs_err:%.3f, rmse:%.3f, >0.5cm:%.2f%%, >0.1cm:%.2f%%, '
            '>1.0cm:%.2f%%, le90:%.3f, grad_err:%.4f') % (
        metric_all[0], metric_all[1], metric_all[2]*100,
        metric_all[3]*100, metric_all[4]*100, metric_all[5], metric_all[6])
    print(info)

                                            