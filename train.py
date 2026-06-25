import argparse
import os
import shutil
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import math
from typing import Tuple
from tqdm import tqdm
from utils.dataset import RSRD
from torch.cuda.amp import GradScaler
from models.loss import MyLoss, LossReg, LossReg2, affine_invariant_global_loss, MSE_normal_loss
from models.structural_losses import CompositeLoss
from torch.utils.data import DataLoader
from models.model import Elevation as ElevationDA3, visualize_encoder_pca
from models.model_dinov2_fb import Elevation as ElevationDinoV2FB
import pickle
from torch.hub import load_state_dict_from_url
import os
from utils.metric import Metric
from utils.experiment import *
import numpy as np
from datetime import datetime
from cardset.dataset import CARDSetDataset, CARDSetDatasetV2Smalldataset
import wandb
import time
import matplotlib.pyplot as plt
from models.reprojection_loss import ReprojectionLoss

from utils.config import parse_args_with_config


os.environ['WANDB_MODE'] = 'online'

now = datetime.now()

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler= None, device="cuda"):
    """
    Loads training checkpoint.

    Args:
        checkpoint_path (str): path to .pt file
        model (torch.nn.Module): model instance
        optimizer (torch.optim.Optimizer, optional)
        device (str): device to map checkpoint to

    Returns:
        start_epoch (int)
        global_step (int)
    """

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

                
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

                                
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("steps", 0)

    print(f"Resumed from epoch {start_epoch}, global step {global_step}")

    return start_epoch, global_step


def get_percentile_bounds(gt, mask, lower_pct: float = 5.0, upper_pct: float = 95.0):
    """
    Compute the visualization range from valid ground truth values.

    Uses the lower and upper percentiles of masked ground truth values to exclude outliers.

    Args:
        gt: torch.Tensor or numpy array of ground truth height values.
        mask: torch.Tensor or numpy array mask where 1 indicates valid values.
        lower_pct: Lower percentile to use for vmin.
        upper_pct: Upper percentile to use for vmax.

    Returns:
        tuple(float, float): (vmin, vmax)
    """
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    valid = gt[mask != 0]
    if valid.size == 0:
        valid = gt.flatten()

    vmin = float(np.percentile(valid, lower_pct))
    vmax = float(np.percentile(valid, upper_pct))
    if vmin == vmax:
        vmin = float(valid.min())
        vmax = float(valid.max())
    return vmin, vmax

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Early stopping to terminate training when validation loss does not improve.

        Parameters:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change in monitored value to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, current_loss):
        """
        Check if training should stop based on current validation loss.

        Parameters:
            current_loss (float): Current epoch's validation loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = current_loss
            return False

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
):
    def lr_lambda(current_step):
                      
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
                            
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def unnormalize(ele_pred, h_min, h_max):
                                                  
                                                                    
    ele_pred = ele_pred * ((h_max - h_min) / 2) + ((h_max + h_min) / 2)
    return ele_pred

def train_regression():
    print("Training with regression loss")
    run = wandb.init(
        entity = "erwan-adonie-njike-ndjongang-cariad",
        project = "RoadHeightFormer",
        name = args.name_run +  str(now.month) + '/' + str(now.day),
        notes = args.notes,
        dir = "/data/rhf/wandb",
                          
                          
        config ={
            "learning_rate" : args.lr,
            "epochs": args.epochs,
            "dataset": args.dataset,
            "trainloader length": len(train_loader),
            "testloader length": len(test_loader),
            "scheduler" : args.scheduler,
            "backbone" : args.backbone,
            "loss_function" : args.loss,
            "Batch_size" : args.batch_size,
    })
    start_epoch = 0
    start_step = 0
    if args.load_pt is not None:
        start_epoch, start_step = load_checkpoint(args.load_pt, model, optimizer, scheduler)
    log_file.write(f"mode: monocular, Dino_type:{args.backbone}, embed_pred_head_input_dim: {args.pred_head_dim}, backbone: {args.backbone}, loss:{args.loss}, batchsize: {args.batch_size}, gradient_weigh: {args.gradient_weight} \n")
    global_step = start_step
    logged_train_static = False
    gt_vmax = [0, 0, 0]
    gt_vmin = [14, 14, 14]
    logged_eval_static = False

    for epoch_idx in tqdm(range(start_epoch, args.epochs)): 
        time_epoch = time.time()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch_idx+1}", unit="batch") as pbar:
            for i, sample in enumerate(train_loader):
                global_step += 1
                start_time = time.time()
                if args.stereo:
                    (imgs_left, imgs_right, ele_gt, ele_mask, proj_index_left, proj_index_right, _) = sample
                    imgs_right, proj_index_right = imgs_right.cuda(), proj_index_right.cuda()
                else:
                    (imgs_left, ele_gt, ele_mask, proj_index_left, _) = sample
                imgs_left, ele_gt, ele_mask, proj_index_left = imgs_left.cuda(), ele_gt.cuda(), ele_mask.cuda(), proj_index_left.cuda()

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    if args.stereo:
                        ele_pred = model(imgs_left, proj_index_left, imgs_right, proj_index_right)
                    else:
                        ele_pred = model(imgs_left, proj_index_left)

                                                                       
                    if i == 0:
                        try:
                            model.eval()
                            with torch.no_grad():
                                _ = model(train_imgs_fixed[:1].cuda(), train_proj_fixed[:1].cuda())
                            model.train()
                            rgb = visualize_encoder_pca(
                                model._last_features,
                                f"/tmp/pca_encoder_epoch{epoch_idx+1}.png",
                            )
                            wandb.log(
                                {"train/encoder_features_pca": wandb.Image(
                                    rgb, caption=f"epoch {epoch_idx+1} (fixed sample 0)")},
                                step=global_step,
                            )
                        except Exception as e:
                            print(f"[pca log] skipped: {e}")

                    loss_all = loss_func(ele_pred, ele_gt, ele_mask)

                                          
                    ele_mask_roi = torch.logical_and(ele_gt > -ele_range*100, ele_gt < ele_range*100)
                    eval_mask = torch.logical_and(ele_mask_roi, ele_mask)
                    eval_mask = eval_mask.bool()
                    with torch.no_grad():
                        mae_l1 = torch.abs(ele_pred.detach()[eval_mask] - ele_gt[eval_mask]).mean()
                    
                                                    
                    if args.normalize:
                                                                   
                        h_min = - ele_range * 100
                        h_max = ele_range * 100                        
                        ele_pred_fixed = unnormalize(ele_pred_fixed, h_min, h_max)
                        print("max and min after normalization:", ele_pred_fixed.max().item(), ele_pred_fixed.min().item())

                                                 
                print("logging step:", global_step, args.summary_freq)
                if global_step % args.summary_freq == 0: 
                    model.eval()
                    with torch.no_grad():
                        ele_pred_fixed = model(train_imgs_fixed.cuda(), train_proj_fixed.cuda())
                    
                        log_dict = {}
                        for s in range(len(fixed_train_indices)):
                            if not logged_train_static:
                                gt_vmin[s], gt_vmax[s] = get_percentile_bounds(
                                    train_gt_fixed[s],
                                    train_mask_fixed[s],
                                    lower_pct=5.0,
                                    upper_pct=95.0,
                                )

                            height_prediction = ele_pred_fixed[s]      
                            combined_img = wandb_combined_image(
                            height_prediction.squeeze(),
                            train_gt_fixed[s],
                            train_mask_fixed[s],
                            train_imgs_fixed[s],
                            caption=f"Combined Visualization of sample {s} at step {global_step}",
                            vmin=gt_vmin[s],
                            vmax=gt_vmax[s],
                            )
                            wandb.log({"train/combined_sample_" + str(s): combined_img}, step=global_step)

                        logged_train_static = True
                    model.train()
                    torch.cuda.empty_cache()
                                                          
                scaler.scale(loss_all).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                                                     
                if args.regression and args.scheduler == 'reduceonplateau':
                    scheduler.step(loss_all.data.item())
                else:
                    scheduler.step()

                wandb.log({"train/lr": scheduler.get_last_lr()[0]}, step = global_step)
                if len(scheduler.get_last_lr()) > 1:
                    wandb.log({"train/lr_encoder": scheduler.get_last_lr()[1]}, step = global_step)
                wandb.log({"train/mae": mae_l1.item()}, step=global_step)
                epoch_active_time = time.time() - start_time
                loss_wandb = loss_all.detach().item()

                if np.isnan(loss_wandb):
                    print('nan loss!')
                    torch.save(model.state_dict(), f"modelbeforebreak{args.name_run}.ckpt")
                    exit()
                print("loss has been logged")
                wandb.log({"loss": loss_wandb}, step = global_step)

                                                                 
                if hasattr(loss_func, '_last_components'):
                    for k, v in loss_func._last_components.items():
                        wandb.log({f"loss/{k}": v}, step=global_step)

                info = 'train--> epoch%2d, lr:%.6f, loss:%.4f' % (epoch_idx+1, optimizer.param_groups[0]['lr'], loss_wandb)
                print(info)


                if global_step % (10*args.summary_freq) == 0:
                    """    loss_data = loss_all.data.item()
                    if np.isnan(loss_data):
                        print('nan loss!')
                        exit() 
                    """
                    log_file.write(info + '\n')
                    log_file.flush()
                    [metric_all, _], eval_loss = test_sample_regression(test_loader, global_step, run, logged_eval_static)
                    wandb.log({"metrics/eval_loss": eval_loss}, step = global_step)
                    wandb.log({"metrics/metric_abs": metric_all[0]}, step = global_step)
                    wandb.log({"metrics/metric_rmse": metric_all[1]}, step = global_step)
                    wandb.log({"metrics/metric_gt05cm": metric_all[2]}, step = global_step)
                                                                               
                    wandb.log({"metrics/abs_err_0.1": metric_all[3]}, step = global_step)
                    wandb.log({"metrics/abs_err_1": metric_all[4]}, step = global_step)
                    wandb.log({"metrics/le90": metric_all[5]}, step = global_step)
                    wandb.log({"metrics/grad_err": metric_all[6]}, step = global_step)

                if global_step % (100 * args.summary_freq) == 0:
                    torch.save({"model": model.state_dict(),
                               "optimizer": optimizer.state_dict(),
                                "epoch": epoch_idx + 1,
                                "steps": global_step}, "{}/checkpoint_epoch{:0>2}_{:0>6}.pt".format(args.logdir, epoch_idx+1, global_step))

                    torch.cuda.empty_cache() 

                    early_stopping(eval_loss)
                    

                    info = 'test:  abs_err:%.3f, rmse:%.3f, >0.5cm:%.2f, grad_error:%.3f eval_loss:%.3f' % (metric_all[0], metric_all[1],metric_all[2]*100, metric_all[-1], eval_loss)
                    log_file.write(info + '\n')
                    log_file.flush()
                    print(info)
            pbar.update(1)
                                            
                                                    
        time_epoch_end = time.time() - time_epoch
        wandb.log({"epoch/epoch_duration": time_epoch_end}, step=global_step)
        wandb.log({"epoch/epoch": epoch_idx+1}, step=global_step)

        last_epoch = epoch_idx + 1

                                               
    final_path = "{}/final_{}_epoch{:0>2}_{:0>6}.pt".format(
        args.logdir, args.name_run.strip() or "run", last_epoch, global_step)
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": last_epoch,
                "steps": global_step}, final_path)
    print(f"[final ckpt] saved {final_path}")
    run.finish()
@make_nograd_func
def test_sample_regression(test_loader, global_step, run, logged_eval_static):
    model.eval()
    eval_loss = 0.0
    gt_vmin = [0, 0, 0]
    gt_vmax = [14, 14, 14]
    h_min = - ele_range*100
    h_max = ele_range*100
    total_error = 0.0
    total_valid_pixels = 0
                                        
    with torch.no_grad():
        ele_pred_fixed = model(eval_imgs_fixed.cuda(), eval_proj_fixed.cuda())
        for s in range(len(fixed_eval_indices)):
            if args.normalize:
                print("undo normalization in visualization of some testing sample")
                ele_pred_fixed = unnormalize(ele_pred_fixed, h_min, h_max)
            
            if not logged_eval_static:
                gt_vmin[s], gt_vmax[s] = get_percentile_bounds(
                    eval_gt_fixed[s],
                    eval_mask_fixed[s],
                    lower_pct=5.0,
                    upper_pct=95.0,
                )

            height_prediction = ele_pred_fixed[s]
            combined_img = wandb_combined_image(
                            height_prediction.squeeze(),
                            eval_gt_fixed[s],
                            eval_mask_fixed[s],
                            eval_imgs_fixed[s],
                            caption=f"Combined Evaluation Visualization at step {global_step}",
                            vmin=gt_vmin[s],
                            vmax=gt_vmax[s],
                            test=True
                            )
            wandb.log({"test/combined_sample_" + str(s): combined_img}, step=global_step)
        logged_eval_static = True
        for i, sample in enumerate(test_loader):
            if args.stereo:
                (imgs_left, imgs_right, ele_gt, ele_mask, proj_index_left, proj_index_right, _) = sample
                imgs_right, proj_index_right = imgs_right.cuda(), proj_index_right.cuda()
            else:
                (imgs_left, ele_gt, ele_mask, proj_index_left, _) = sample
            imgs_left, ele_gt, ele_mask, proj_index_left = imgs_left.cuda(), ele_gt.cuda(), ele_mask.cuda(), proj_index_left.cuda()
            
            with torch.cuda.amp.autocast(dtype=torch.float16):

                if args.stereo:
                    ele_pred = model(imgs_left, proj_index_left, imgs_right, proj_index_right)
                else:
                    ele_pred = model(imgs_left, proj_index_left)
                                                                                           
                                                                                
                if args.normalize:
                    print("undo normalization in testing")
                    h_min = - ele_range*100
                    h_max = ele_range*100
                    ele_pred = unnormalize(ele_pred, h_min, h_max)

                metric.compute(ele_pred, ele_gt, ele_mask)
                                                                 

                abs_error = (torch.abs(ele_pred - ele_gt)) * ele_mask
                valid_count = ele_mask.sum().clamp(min=1)
                abs_error = abs_error.sum() / valid_count                                     
                total_error += abs_error.item()
                total_valid_pixels += ele_mask.sum().item()
            del ele_pred, imgs_left, ele_gt, ele_mask, proj_index_left

    model.train()
    torch.cuda.empty_cache()
    metric_values = metric.get_metric()
    metric.clear()
    eval_loss = total_error/len(test_loader)                        
    return metric_values, eval_loss


def train():
    print("Train classificationmodel")
    run = wandb.init(
        entity = "erwan-adonie-njike-ndjongang-cariad",
        project = "RoadHeightFormer",
        name = args.name_run +  str(now.month) + '/' + str(now.day),
                       
                         
        notes = args.notes,
        config ={
            "learning_rate" : args.lr,
            "epochs": args.epochs,
            "dataset": args.dataset,
            "trainloader length": len(train_loader),
            "testloader length": len(test_loader),
            "scheduler" : args.scheduler,
            "backbone" : args.backbone,
            "loss_function" : args.loss,
            "Batch_size" : args.batch_size,
    })
    global_step = 0
    logged_train_static = False
    gt_vmax = [0, 0, 0]
    gt_vmin = [14, 14, 14]
    logged_eval_static = False
    for epoch_idx in tqdm(range(args.epochs)):
        time_epoch = time.time()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch_idx+1}", unit="batch") as pbar:
            for i, sample in enumerate(train_loader):
                global_step += 1
                start_time = time.time()
                if args.stereo:
                    (imgs_left, imgs_right, ele_gt, ele_mask, proj_index_left, proj_index_right, _) = sample
                    imgs_right, proj_index_right = imgs_right.cuda(), proj_index_right.cuda()
                else:
                    (imgs_left, ele_gt, ele_mask, proj_index_left, _) = sample
                    """ if args.reprojection_loss:
                        (_, road_info, neighbours, extrinsics, intrinsics, points_data) = sample
                        gt_depth = _project_pts_to_depth(points_data.numpy(), intrinsics.numpy(), HW=(952, 528))
                        ground_info, neighbours, extrinsics, intrinsics, gt_depth = road_info.cuda(), neighbours.cuda(), extrinsics.cuda(), intrinsics.cuda(), gt_depth.cuda()
                        extrinsics_next = neighbours[-1]['extrinsics']
                        extrinsics_previous = neighbours[0]['extrinsics']
                        extrinsics_next, extrinsics_previous = extrinsics_next.cuda(), extrinsics_previous.cuda()
                        I_previous = neighbours[0]['rgb']
                        I_next = neighbours[-1]['rgb'] """
                        
                imgs_left, ele_gt, ele_mask, proj_index_left = imgs_left.cuda(), ele_gt.cuda(), ele_mask.cuda(), proj_index_left.cuda()


                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    if args.stereo:
                        ele_pred = model(imgs_left, proj_index_left, imgs_right, proj_index_right)
                    else:
                        ele_pred = model(imgs_left, proj_index_left)
                        print("train ele pred shape:", ele_pred.shape)
                    if i == 0:
                        try:
                            model.eval()
                            with torch.no_grad():
                                _ = model(train_imgs_fixed[:1].cuda(), train_proj_fixed[:1].cuda())
                            model.train()
                            rgb = visualize_encoder_pca(
                                model._last_features,
                                f"/tmp/pca_encoder_epoch{epoch_idx+1}.png",
                            )
                            wandb.log(
                                {"train/encoder_features_pca": wandb.Image(
                                    rgb, caption=f"epoch {epoch_idx+1} (fixed sample 0)")},
                                step=global_step,
                            )
                        except Exception as e:
                            print(f"[pca log] skipped: {e}")
                        finally:
                            model.train()

                    loss_all = loss_func(ele_pred, ele_gt, ele_mask)
                                          
                    ele_mask_roi = torch.logical_and(ele_gt > -ele_range, ele_gt < ele_range)
                    eval_mask = torch.logical_and(ele_mask_roi, ele_mask)
                    height_prediction =  F.softmax(ele_pred, dim= 1)
                    
                    height_prediction = torch.sum(height_prediction * model.ele_values,dim=1)
                    with torch.no_grad():
                        mae_l1 = (torch.abs(height_prediction[eval_mask] - ele_gt[eval_mask])).mean()
                                                                                                                                                                            

                print("logging step:", global_step, args.summary_freq)
                if global_step % args.summary_freq == 0: 
                    log_dict = {}
                    model.eval()
                    with torch.no_grad():
                        ele_pred_fixed = model(train_imgs_fixed.cuda(), train_proj_fixed.cuda())
                    for s in range(len(fixed_train_indices)):
                        if not logged_train_static:
                            gt_vmin[s], gt_vmax[s] = get_percentile_bounds(
                                train_gt_fixed[s],
                                train_mask_fixed[s],
                                lower_pct=5.0,
                                upper_pct=95.0,
                            )

                        height_prediction =  F.softmax(ele_pred_fixed[s], dim=0)
                        height_prediction = torch.sum(height_prediction * model.ele_values[0],dim=0)
                        combined_img = wandb_combined_image(
                        height_prediction.squeeze(),
                        train_gt_fixed[s],
                        train_mask_fixed[s],
                        train_imgs_fixed[s],
                        caption=f"Combined Visualization of sample {s} at step {global_step}",
                        vmin=gt_vmin[s],
                        vmax=gt_vmax[s],
                        )
                        wandb.log({"train/combined_sample_" + str(s): combined_img}, step=global_step)
                    model.train()

                    logged_train_static = True
                                                          
                scaler.scale(loss_all).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                epoch_active_time = time.time() - start_time
                wandb.log({"train/lr": scheduler.get_last_lr()[0]}, step = global_step)
                if len(scheduler.get_last_lr()) > 1:
                    wandb.log({"train/lr_encoder": scheduler.get_last_lr()[1]}, step = global_step)
                wandb.log({"train/mae": mae_l1.item()}, step=global_step)
                loss_wandb = loss_all.data.item()
                if np.isnan(loss_wandb):
                    print('nan loss!')
                    exit()
                print("loss has been logged")
                wandb.log({"loss": loss_wandb}, step = global_step)

                if global_step % args.summary_freq == 0:
                    loss_data = loss_all.data.item()
                    if np.isnan(loss_data):
                        print('nan loss!')
                        exit()
                    info = 'train--> epoch%2d, lr:%.6f, loss:%.4f' % (epoch_idx+1, optimizer.param_groups[0]['lr'], loss_data)
                    log_file.write(info + '\n')
                    log_file.flush()
                    print(info)

                if global_step % (10*args.summary_freq) == 0:
                                                                                                                                          

                    [metric_all, _], eval_loss = test_sample(test_loader, global_step, run, logged_eval_static)
                    wandb.log({"metrics/eval_loss": eval_loss}, step = global_step)
                    wandb.log({"metrics/metric_abs": metric_all[0]}, step = global_step)
                    wandb.log({"metrics/metric_rmse": metric_all[1]}, step = global_step)
                    wandb.log({"metrics/metric_gt05cm": metric_all[2]}, step = global_step)
                    wandb.log({"metrics/abs_err_0.1": metric_all[3]}, step = global_step)
                    wandb.log({"metrics/abs_err_1": metric_all[4]}, step = global_step)
                    wandb.log({"metrics/le90": metric_all[5]}, step = global_step)
                    wandb.log({"metrics/grad_err": metric_all[6]}, step = global_step)
                    early_stopping(eval_loss)
                    

                    info = 'test:    abs_err:%.3f, rmse:%.3f, >0.5cm:%.2f, eval_loss:%.3f' % (metric_all[0], metric_all[1], metric_all[2]*100, eval_loss)
                    log_file.write(info + '\n')
                    log_file.flush()
                    print(info)
                
                if global_step % (300 * args.summary_freq) == 0:
                    torch.save({"model": model.state_dict(),
                               "optimizer": optimizer.state_dict(),
                                "epoch": epoch_idx + 1,
                                "steps": global_step}, "{}/checkpoint_epoch{:0>2}_{:0>6}.pt".format(args.logdir, epoch_idx+1, global_step))

                    torch.cuda.empty_cache() 
                epoch_passiv_time = time.time() - epoch_active_time
                                                                                                       
            pbar.update(1)
            time_epoch_end = time.time() - time_epoch
            wandb.log({"epoch/epoch_duration": time_epoch_end}, step=global_step)
            wandb.log({"epoch/epoch": epoch_idx+1}, step=global_step)
            if early_stopping.should_stop:
                print("Early stopping triggered!")
                break

                                                                     
    final_path = "{}/final_{}_epoch{:0>2}_{:0>6}.pt".format(
        args.logdir, args.name_run.strip() or "run", epoch_idx + 1, global_step)
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_idx + 1,
                "steps": global_step}, final_path)
    print(f"[final ckpt] saved {final_path}")
    run.finish()
@make_nograd_func
def test_sample(test_loader, global_step, run, logged_eval_static=False):
    model.eval()
    eval_loss = 0.0
    gt_vmin = [0, 0, 0]
    gt_vmax = [14, 14, 14]
    h_min = - ele_range*100
    h_max = ele_range*100
                                        
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        ele_pred_fixed = model(eval_imgs_fixed.cuda(), eval_proj_fixed.cuda())
    total_error = 0.0
    total_valid_pixels = 0

    for s in range(len(fixed_eval_indices)):
        if args.normalize:
            print("undo normalization in visualization of some testing sample")
            ele_pred_fixed = ele_pred_fixed * ((h_max - h_min) / 2) + ((h_max + h_min) / 2)
        
        if not logged_eval_static:
            gt_vmin[s], gt_vmax[s] = get_percentile_bounds(
                eval_gt_fixed[s],
                eval_mask_fixed[s],
                lower_pct=5.0,
                upper_pct=95.0,
            )

        height_prediction = ele_pred_fixed[s]
        combined_img = wandb_combined_image(
                        height_prediction.squeeze(),
                        eval_gt_fixed[s],
                        eval_mask_fixed[s],
                        eval_imgs_fixed[s],
                        caption=f"Combined Evaluation Visualization at step {global_step}",
                        vmin=gt_vmin[s],
                        vmax=gt_vmax[s],
                        test=True
                        )
        wandb.log({"test/combined_sample_" + str(s): combined_img}, step=global_step)

    for i, sample in enumerate(test_loader):
        if args.stereo:
            (imgs_left, imgs_right, ele_gt, ele_mask, proj_index_left, proj_index_right, _) = sample
            imgs_right, proj_index_right = imgs_right.cuda(), proj_index_right.cuda()
        else:
            (imgs_left, ele_gt, ele_mask, proj_index_left, _) = sample
        imgs_left, ele_gt, ele_mask, proj_index_left = imgs_left.cuda(), ele_gt.cuda(), ele_mask.cuda(), proj_index_left.cuda()
        
        with torch.cuda.amp.autocast(dtype=torch.float16):

            if args.stereo:
                ele_pred = model(imgs_left, proj_index_left, imgs_right, proj_index_right)
            else:
                ele_pred = model(imgs_left, proj_index_left)
            metric.compute(ele_pred, ele_gt, ele_mask)
                                                          
                                                               
            abs_error = torch.abs(
                ele_pred - ele_gt
            ) * ele_mask

            valid_count = ele_mask.sum().clamp(min=1)
            abs_error = abs_error.sum() / valid_count                                     
            total_error += abs_error.item()
            total_valid_pixels += ele_mask.sum().item()
            del ele_pred, imgs_left, ele_gt, ele_mask, proj_index_left


    model.train()
    torch.cuda.empty_cache()
    metric_values = metric.get_metric()
    metric.clear()
    eval_loss = total_error/len(test_loader)

    return metric_values, eval_loss

def get_pointcloud_from_heightmap(heightmap, centers):
    '''
    Parameters: heighmapmap shape B, Z, X
    centers: centers position B, Z, X, 2

    return: a pointcloud B, X, Y, Z
    '''
                            

    heightmap = heightmap.unsqueeze(dim=-1)
    result = torch.cat((centers, heightmap), dim=-1)
    result = result.permute(-1, -2)
    
    return result
    

def _project_pts_to_depth(pts_cam: np.ndarray, K: np.ndarray, HW: Tuple[int,int]) -> np.ndarray:
    H,W = HW; dep = np.zeros((H,W), np.float32)
    if pts_cam is None or len(pts_cam)==0: return dep
    z = pts_cam[:,2]; ok = z > 0
    if not np.any(ok): return dep
    x,y,z = pts_cam[ok,0], pts_cam[ok,1], pts_cam[ok,2]
    u = (K[0,0]*x/z) + K[0,2]; v = (K[1,1]*y/z) + K[1,2]
    m = (u>=0)&(u<W)&(v>=0)&(v<H)
    if not np.any(m): return dep
    u,v,z = u[m], v[m], z[m]
    order = np.argsort(z)[::-1]
    u,v,z = u[order], v[order], z[order]
    dep[v.astype(int), u.astype(int)] = z
    return dep

def wandb_error_map(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    caption: str = "",
    cmap: str = "RdBu_r",
    vmin: float = None,
    vmax: float = None,
) -> wandb.Image:
    """
    Compute per-cell error (pred - gt) and return as wandb.Image.

    Args:
        pred: torch.Tensor (H, W) prediction
        gt: torch.Tensor (H, W) ground truth
        mask: torch.Tensor (H, W), 0 = invalid
        caption: image caption
        cmap: diverging colormap 
        vmin/vmax: optional fixed range for color normalization (recommended)

    Returns:
        wandb.Image
    """
                      
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

                           
    error = pred - gt
    error = np.ma.masked_where(mask == 0, error)

                  
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(error, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Error (cm)")
    fig.tight_layout()

    return wandb.Image(fig, caption=caption)

def wandb_combined_image(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    img: torch.Tensor,
    caption: str = "",
    cmap: str = "plasma",
    vmin: float = None,
    vmax: float = None,
    test = False
) -> wandb.Image:
    """
    Combine prediction, ground truth, and error map into a single image.

    Args:
        pred: torch.Tensor (H, W) prediction.
        gt: torch.Tensor (H, W) ground truth.
        mask: torch.Tensor (H, W), 0 = invalid.
        caption: Image caption.
        cmap: Colormap for the heightmaps.
        vmin/vmax: Optional fixed range for color normalization.

    Returns:
        wandb.Image: Combined image for logging.
    """
                                      
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    img = img.detach().cpu()
    img = denormalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    img = img.permute(1, 2, 0).numpy()

                                  
    if test == False:
        pred = np.ma.masked_where(mask == 0, pred)    
    gt = np.ma.masked_where(mask == 0, gt)
    error = np.ma.masked_where(mask == 0, pred - gt)

                                                                                
    error_values = error.compressed() if np.ma.is_masked(error) else error.flatten()
    if error_values.size > 0:
        error_range = np.percentile(np.abs(error_values), 95.0)
        error_vmin, error_vmax = -float(error_range), float(error_range)
        if error_vmin == error_vmax:
            error_vmin = float(error_values.min())
            error_vmax = float(error_values.max())
    else:
        error_vmin, error_vmax = -1.0, 1.0

                               
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

                
    im_pred = axes[0].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Prediction")
    axes[0].axis("off")
    fig.colorbar(im_pred, ax=axes[0], fraction=0.046, pad=0.04)

                  
    im_gt = axes[1].imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    fig.colorbar(im_gt, ax=axes[1], fraction=0.046, pad=0.04)

               
    im_error = axes[2].imshow(error, cmap="RdBu_r", vmin=error_vmin, vmax=error_vmax)
    axes[2].set_title("Error Map")
    axes[2].axis("off")
    fig.colorbar(im_error, ax=axes[2], fraction=0.046, pad=0.04, label="Error (cm)")

    image = axes[3].imshow(img)
    axes[3].set_title("GT image")
    axes[3].axis("off")


    fig.tight_layout()
                                   
    return wandb.Image(fig, caption=caption)

def wandb_heightmap_image(height_map, mask, caption, cmap="plasma", vmin = None, vmax = None):
    """
    height_map, mask: torch.Tensor (H, W)
    """

    print("....saving file wandb")
    height_map = height_map.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()


    height_map = np.ma.masked_where(mask == 0, height_map)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(height_map, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cm")
    fig.tight_layout()
    
    return wandb.Image(fig, caption=caption)

def wandb_rgb_image(img, caption):
    img = img.detach().cpu().permute(1, 2, 0).numpy()
    return wandb.Image(img, caption=caption)

def get_fixed_samples(dataset, indices):
    samples = [dataset[i] for i in indices]
    print("fetch from visual")
    imgs = torch.stack([s[0] for s in samples])
    ele_gt = torch.stack([s[1] for s in samples])
    ele_mask = torch.stack([s[2] for s in samples])
    proj_idx = torch.stack([s[3] for s in samples])

    return imgs, ele_gt, ele_mask, proj_idx

def denormalize(img, mean, std):
    """
    Denormalize a normalized image.

    Args:
        img: torch.Tensor (C, H, W) normalized image.
        mean: List of mean values for each channel.
        std: List of standard deviation values for each channel.

    Returns:
        torch.Tensor: Denormalized image.
    """
    mean = torch.tensor(mean).view(3, 1, 1)                        
    std = torch.tensor(std).view(3, 1, 1)                          
    return img * std + mean

def sum_absolute_error(pred, gt):
    """
    Compute the Sum of Absolute Error (SAE) between two elevation maps.

    Parameters:
        pred (torch.Tensor): Predicted elevation map of shape (1, 164, 64).
        gt (torch.Tensor): Ground truth elevation map of shape (1, 164, 64).

    Returns:
        float: Sum of absolute error.
    """
                                                         
    pred = pred.to(dtype=torch.float32)
    gt = gt.to(dtype=torch.float32)

                 
    assert pred.shape == gt.shape, f"got {pred.shape} and {gt.shape}"

                 
    sae = torch.sum(torch.abs(pred - gt)).item()
    return sae


def analyze_backbone_frozen_status(model):
    """
    Analyze which parts of the model are frozen.
    Useful for understanding training behavior.
    """
    print("\n" + "=" * 100)
    print("BACKBONE FROZEN STATUS ANALYSIS")
    print("=" * 100)
    
    frozen_modules = []
    trainable_modules = []
    
    for name, module in model.named_modules():
        module_trainable = any(p.requires_grad for p in module.parameters())
        module_frozen = any(not p.requires_grad for p in module.parameters())
        
        if module_trainable and module_frozen:
            status = "MIXED"
            trainable_modules.append(name)
        elif module_trainable:
            status = "TRAINABLE"
            trainable_modules.append(name)
        else:
            status = "FROZEN"
            frozen_modules.append(name)
        
        param_count = sum(p.numel() for p in module.parameters())

    log_file.write(f"\Trainable Modules: {len(trainable_modules)}")
    for m in trainable_modules:
        log_file.write(f"  - {m}")
    
    log_file.write(f"\nFrozen Modules: {len(frozen_modules)}")
    for m in frozen_modules:
        log_file.write(f"  - {m}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RoadBEV: Road Surface Reconstruction in Bird\'s Eye View')
    parser.add_argument('--dataset', help='dataset to use: add it to wandb runs')
    parser.add_argument('--stereo', action='store_true', help='if yes, use RoadBEV-stereo; otherwise, RoadBEV-mono')
    parser.add_argument('--cla_res', type=float, default=0.5, help='class resolution for elevation classification')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='maximum learning rate')
    parser.add_argument('--lr_encoder', type=float, default=1e-5, help='peak LR for the encoder param group (only relevant when --train_encoder is set; encoder is frozen otherwise).')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--logdir', default='/data/rhf/checkpoints/', help='the directory to save logs and checkpoints')
    parser.add_argument('--loadckpt', default=None, help='load the weights from a specific checkpoint')
    parser.add_argument('--summary_freq', type=int, default=10, help='summary_freq')
    parser.add_argument('--seed', type=int, default=307, metavar='S', help='random seed')
    parser.add_argument('--regression', action='store_true', help='regression or classification')
    parser.add_argument('--backbone',default='efficientnet', help='Use DepthAnything3 backbone or EfficientNet')
    parser.add_argument('--gradient_weight', type=float, default=0.01, help='weight for gradient loss in regression')
    parser.add_argument('--notes', type=str, default='', help='notes for wandb run')
    parser.add_argument('--scheduler', type=str, default='onecycle', help='type of lr scheduler to use: onecycle or reduceonplateau')
    parser.add_argument('--loss', type=str, default='L1', help='type of loss to use if regression: L1, gaussian NLL')
    parser.add_argument('--normalize', action='store_true', help='disable normalization')
    parser.add_argument('--name_run', type=str, default= ' ', help='give the name of the wandb run')
    parser.add_argument('--pred_head_dim', type=int, default=128, help='define the bottleneck between the transformer encoder and the CNN prediction head')
    parser.add_argument('--preprocessed', action='store_true', help='if yes, the dataloader will load preprocessed data')
    parser.add_argument('--load_pt', default=None, help='load weights, optimizer, start_idx to resume run')
    parser.add_argument('--dino', default="small", help='ViT encoder size')
    parser.add_argument('--clamp_gt', action='store_true', help='if set, clamp GT elevation values to [-y_range*100, y_range*100] cm in the dataloader (in addition to the existing ROI mask filtering)')
    parser.add_argument('--crop_to_road', action='store_true', help='if set, the dataloader crops each image to a bbox, resizes down to size which are % 14 to match the dino patchsize, and adjusts the intrinsic / voxel_uv accordingly.')
    parser.add_argument('--train_encoder', action='store_true', help='if set, the DepthAnything3 backbone runs without torch.no_grad() so its weights are updated during training (default: encoder is frozen).')
    parser.add_argument('--w_pixel', type=float, default=0.3, help='composite loss: pixel term weight')
    parser.add_argument('--w_gradient', type=float, default=1.0, help='composite loss: gradient term weight')
    parser.add_argument('--w_structure', type=float, default=0.0, help='composite loss: SSIM-like structural term weight')
    parser.add_argument('--w_normal', type=float, default=1.0, help='composite loss: surface-normal term weight')
    parser.add_argument('--w_smoothness', type=float, default=0.0, help='composite loss: edge-aware smoothness term weight')
    parser.add_argument('--dinov2_layers', type=int, nargs='+', default=[5, 7, 9, 11], help='which DINOv2 transformer block indices to extract intermediate features from (used by --backbone DINOv2_fb).')
    parser.add_argument('--upsampler_kind', type=str, default='patch2feature', choices=['patch2feature', 'dino'], help='which upsampler to plug after the DINOv2 encoder: patch2feature (DPT-style multi-stage fusion) or dino (DinoUpsampler).')
    parser.add_argument('--pixel_type', type=str, default='MSE', choices=['L1', 'MSE'], help='pixel-term loss inside CompositeLoss: L1 or MSE.')

                                
    args = parse_args_with_config()
    print(f"[config] file:       {args.config}")
    print(f"[config] dataset:    {args.dataset}")
    print(f"[config] backbone:   {args.backbone}")
    print(f"[config] batch_size: {args.batch_size}")
    print(f"[config] epochs:     {args.epochs}")
                                
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True
                                             
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print("normalize", args.normalize)

    if args.stereo:
        args.down_scale = 2
        print('training RoadBEV-stereo!')
    else:
        args.down_scale = 4
        print('training RoadBEV-mono!')

                         
    voxel_kwargs = {'y_range': getattr(args, 'y_range', None),
                    'num_grids_y': getattr(args, 'num_grids_y', None)}

    if 'RSRD' in args.dataset:
        train_set = RSRD(training=True, stereo=args.stereo, down_scale=args.down_scale, backbone=args.backbone)
        test_set = RSRD(training=False, stereo=args.stereo, down_scale=args.down_scale, backbone=args.backbone)

    elif 'CARDSetV2Small' in args.dataset:
        test_set = CARDSetDatasetV2Smalldataset(root_dir='CARDSet/CARD_nice', mode='test', down_scale=args.down_scale, clamp_gt=args.clamp_gt, crop_to_road=args.crop_to_road, **voxel_kwargs)
        train_set = CARDSetDatasetV2Smalldataset(root_dir='CARDSet/CARD_nice', mode='test', down_scale=args.down_scale, clamp_gt=args.clamp_gt, crop_to_road=args.crop_to_road, **voxel_kwargs)
                                                                                   
        args.batch_size = 1
        args.summary_freq = 1
        args.epochs = 20

    elif 'CARDSet_y04_g40_square' in args.dataset:
        train_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', split_file='/data/rhf/train_dataset_y0.4_g40_square.txt', mode='train', down_scale=args.down_scale, preprocessed_data = args.preprocessed, augmentation = False, clamp_gt=args.clamp_gt, crop_to_road=args.crop_to_road, **voxel_kwargs)
        test_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', split_file='/data/rhf/val_dataset_y0.4_g40_square.txt', mode='test', down_scale=args.down_scale, preprocessed_data = args.preprocessed, augmentation = False, clamp_gt=args.clamp_gt, crop_to_road=args.crop_to_road, **voxel_kwargs)
        train_set.preprocessed_dir = '/data/rhf/train_preprocessed_data_y0.4_g40_square'
        test_set.preprocessed_dir = '/data/rhf/val_preprocessed_data_y0.4_g40_square'
    elif 'CARDSetSmall_cropped' in args.dataset:
        train_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', split_file='/data/rhf/train_small_dataset_thesis_cropped.txt', mode='train', down_scale=args.down_scale, preprocessed_data = args.preprocessed, augmentation = False, clamp_gt=args.clamp_gt, crop_to_road=args.crop_to_road, **voxel_kwargs)
        test_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', split_file='/data/rhf/val_small_dataset_thesis_cropped.txt', mode='test', down_scale=args.down_scale, preprocessed_data = args.preprocessed, augmentation = False, clamp_gt=args.clamp_gt, crop_to_road=args.crop_to_road, **voxel_kwargs)
        train_set.preprocessed_dir = '/data/rhf/train_preprocessed_small_data_thesis_cropped'
        test_set.preprocessed_dir = '/data/rhf/val_preprocessed_small_data_thesis_cropped'
    elif 'CARDSetSmall' in args.dataset:
        train_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', split_file='/data/rhf/train_small_dataset_thesis.txt', mode='train', down_scale=args.down_scale, preprocessed_data = args.preprocessed, augmentation = False, clamp_gt=args.clamp_gt, crop_to_road=args.crop_to_road, **voxel_kwargs)
        test_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', split_file='/data/rhf/val_small_dataset_thesis.txt', mode='test', down_scale=args.down_scale, preprocessed_data = args.preprocessed, augmentation = False, clamp_gt=args.clamp_gt, crop_to_road=args.crop_to_road, **voxel_kwargs)
        train_set.preprocessed_dir = '/data/rhf/train_preprocessed_small_data_thesis'
        test_set.preprocessed_dir = '/data/rhf/val_preprocessed_small_data_thesis'

    elif 'CARDSet' in args.dataset:
        train_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', split_file='/data/T7/cariad dataset/train_all_data_clean_NN_RHF.txt', mode='train', down_scale=args.down_scale, preprocessed_data = args.preprocessed, clamp_gt=args.clamp_gt, crop_to_road=args.crop_to_road, **voxel_kwargs)
        test_set = CARDSetDataset(root_dir='/data/T7/cariad dataset', split_file='/data/T7/cariad dataset/val_all_data_clean_NN_RHF.txt', mode='test', down_scale=args.down_scale, preprocessed_data = args.preprocessed, clamp_gt=args.clamp_gt, crop_to_road=args.crop_to_road, **voxel_kwargs)
        train_set.preprocessed_dir = '/data/rhf/train_preprocessed_data'
        test_set.preprocessed_dir = '/data/rhf/val_preprocessed_data'
    else:
        print('unknown dataset!')
        exit(0)

    print(f"[dataset] preprocessed       : {args.preprocessed}")
    print(f"[dataset] train preprocessed_dir: {getattr(train_set, 'preprocessed_dir', '<not set>')}")
    print(f"[dataset] test  preprocessed_dir: {getattr(test_set,  'preprocessed_dir', '<not set>')}")

                                                                              
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=False)

                                                                                                                  
    test_loader = DataLoader(test_set, 1, shuffle=False, num_workers=8, drop_last=False, pin_memory=False)
    print('dataset size - train:%d, test:%d' % (len(train_loader), len(test_loader)))

                                 
    fixed_train_indices = [0, 1, 2]  
    fixed_eval_indices  = [0, 1, 2]      

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_imgs_fixed, train_gt_fixed, train_mask_fixed, train_proj_fixed = get_fixed_samples(train_loader.dataset, fixed_train_indices)

    eval_imgs_fixed, eval_gt_fixed, eval_mask_fixed, eval_proj_fixed = get_fixed_samples(test_loader.dataset, fixed_eval_indices)
    
                      
    ele_range = train_set.y_range
    voxel_ele_res = train_set.grid_res[1]
    num_grids = [train_set.num_grids_x, train_set.num_grids_y, train_set.num_grids_z]
    hori_centers = train_set.hori_centers.to(device)                         


    Elevation = ElevationDinoV2FB if 'DINOv2_fb' in args.backbone else ElevationDA3
    extra_kwargs = {}
    if 'DINOv2_fb' in args.backbone:
        extra_kwargs['dinov2_layers'] = tuple(args.dinov2_layers)
        extra_kwargs['upsampler_kind'] = args.upsampler_kind
    model = Elevation(args.stereo, num_grids, ele_range, args.cla_res, args.regression, args.backbone, args.normalize, args.pred_head_dim, train_encoder=args.train_encoder, **extra_kwargs).cuda()             
    early_stopping = EarlyStopping(patience=300, min_delta=0.001)
    print('num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(model)

                                      
    global_step = 0
    last_epoch = 0
    model.train()
                                        

    if not args.train_encoder:
        for param in model.feature_extraction.parameters():
            param.requires_grad = False
        model.feature_extraction.eval()
    
    aloss = ReprojectionLoss((952, 518)).cuda()

    if args.regression:
        if args.loss == 'L1':
            loss_func = LossReg(ele_range, args.normalize, 'L1').cuda()
        elif args.loss == 'scale_affine':
            loss_func = affine_invariant_global_loss()
        elif args.loss == 'MSE':
            loss_func = LossReg(ele_range, args.normalize, 'MSE').cuda()
        elif args.loss == 'lpips':
            loss_func = LossReg(ele_range, args.normalize, 'lpips').cuda()
        elif args.loss == 'composite':
            loss_func = CompositeLoss(
                ele_range, hori_centers=hori_centers, normalize=args.normalize,
                pixel_type=args.pixel_type,
                w_pixel=args.w_pixel,
                w_gradient=args.w_gradient,
                w_structure=args.w_structure,
                w_normal=args.w_normal,
                w_smoothness=args.w_smoothness,
            ).cuda()
        else:
            loss_func = MSE_normal_loss(ele_range, hori_centers=hori_centers, normalize=args.normalize).cuda()
    else:
        loss_func = MyLoss(ele_range, voxel_ele_res, args.cla_res).cuda()
    metric = Metric(ele_range, train_set.num_grids_z, distance_wise=False)

    if args.backbone == 'efficientnet':
        url = 'https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth'
        try:
            weights = load_state_dict_from_url(url, progress=True)
        except:
            print('please manually download pretrained weights at:', url)
            exit(0)

        weights_new = {}
        target_keys = ['features.0', 'features.1', 'features.2', 'features.3', 'features.4']
        for key, value in weights.items():
            if any(k in key for k in target_keys):
                weights_new[key.replace('features.', 'l')] = value
        model.feature_extraction.load_state_dict(weights_new, strict=False)

    
    """ if args.loadckpt is not None:
        # load the checkpoint file specified by args.loadckpt, check the log_file text to make sure the config are same before loading
        
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt)
        model.load_state_dict(state_dict, strict=True) """
 
    scaler = GradScaler()
    print("model_state", model.feature_extraction.training)
    lr_encoder = args.lr_encoder
    encoder_params = list(model.feature_extraction.parameters())
    decoder_params = [param for name, param in model.named_parameters() if 'feature_extraction' not in name]
    print(f"number of decoder parameters: {sum(p.numel() for p in decoder_params)} vs number of parameter{sum(p.numel() for p in model.parameters())} vs number of encoder param {sum(p.numel() for p in encoder_params)}")

    optimizer = optim.AdamW([{"params":decoder_params, "lr":args.lr, "betas":(0.9, 0.999), "weight_decay":1e-4},
                             {"params":encoder_params, "lr":lr_encoder, "betas":(0.9, 0.999), "weight_decay":1e-4}])


    if args.scheduler == "cosine" and args.loadckpt is None:
        steps_per_epoch = len(train_loader)

        num_training_steps = args.epochs * steps_per_epoch
        num_warmup_steps = int(0.1 * num_training_steps)             
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_training_steps,
        )
    
    elif args.scheduler == 'reduceonplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    else: 
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[args.lr, lr_encoder], epochs=args.epochs, pct_start=0.1,
                                                        three_phase=False,
                                                        div_factor=20, anneal_strategy='linear',
                                                        steps_per_epoch=len(train_loader))
    

    args.logdir = os.path.join(args.logdir, datetime.utcnow().strftime('%Y%m%d%H%M%S'))
    print('logging dir:', args.logdir)
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs("end_models", exist_ok= True)
    shutil.copy('./cardset/dataset.py', os.path.join(args.logdir, 'dataset.py'))
    shutil.copy('./models/model.py', os.path.join(args.logdir, 'model.py'))
    shutil.copy('./models/efficientnet.py', os.path.join(args.logdir, 'efficientnet.py'))
    shutil.copy('./models/ele_head.py', os.path.join(args.logdir, 'ele_head.py'))
    shutil.copy('./models/patch2feature.py', os.path.join(args.logdir, 'patch2feature.py'))
    shutil.copy('train.py', os.path.join(args.logdir, 'train.py'))
                                                                    
    log_file = open(os.path.join(args.logdir, 'log.txt'), 'a')
    analyze_backbone_frozen_status(model)
    if args.regression:
             
        train_regression()
                                
                                                       
    else:
        train()


