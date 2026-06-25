import numpy as np
from utils.experiment import make_nograd_func
import torch
import matplotlib.pyplot as plt
from cardset.dataset import CARDSetDataset, CARDSetDatasetV2Smalldataset

class Metric():
    def __init__(self, ele_range, num_grids_z, distance_wise=False):
        self.ele_range = ele_range*100
                                     
                                                               
        self.metric_all = np.zeros(7,)
        self.count_all = 0

                                                             
        self.distance_wise = distance_wise
        self.intervals = 11                                     
        self.num_intervals = int(num_grids_z/self.intervals)+1
        self.metric_wise = np.zeros((self.num_intervals, 3))
        self.count_wise = np.zeros(self.num_intervals)

    @make_nograd_func
    def clear(self):
        self.count_all = 0
        self.metric_all *= 0
        self.count_wise *= 0
        self.metric_wise *= 0

    @make_nograd_func
    def plot_depthwise(self, metric_depthwise):
        plt.figure()
        plt.subplot(121)
        plt.plot(np.flip(metric_depthwise[:, 0]), marker='*')
        plt.title('Abs_err')
        plt.subplot(122)
        plt.plot(np.flip(metric_depthwise[:, 1]), marker='*')
        plt.title('RMSE')
        plt.show()

    @make_nograd_func
    def get_metric(self):
                                    
        metric_all = self.metric_all / self.count_all
        if self.distance_wise:
            metric_wise = self.metric_wise / self.count_wise.reshape(-1, 1)
            return [metric_all, metric_wise]
        else:
            return [metric_all, None]

    @make_nograd_func
    def compute_values(self, ele_gt, ele_pred):
        abs_err = torch.abs(ele_gt - ele_pred)
        rmse = (ele_gt - ele_pred) ** 2
        rmse = torch.sqrt(rmse.mean())

        err_mask = abs_err > 0.5
        ratio_thresh = torch.mean(err_mask.float())

                  
        epsilon = 1e-6                               
                                                                                                              

        abs_err_01 = torch.mean((abs_err > 0.1).float())                               
        abs_err_1 = torch.mean((abs_err > 1.0).float())                                

                              
        le90 = torch.quantile(abs_err, 0.9)                                     

                        
        grad_pred_x = torch.abs(ele_pred[:, 1:] - ele_pred[:, :-1])                           
        grad_pred_y = torch.abs(ele_pred[1:, :] - ele_pred[:-1, :])                           
        grad_gt_x = torch.abs(ele_gt[:, 1:] - ele_gt[:, :-1])                                 
        grad_gt_y = torch.abs(ele_gt[1:, :] - ele_gt[:-1, :])                                 
        grad_err_x = torch.abs(grad_pred_x - grad_gt_x).mean()                                      
        grad_err_y = torch.abs(grad_pred_y - grad_gt_y).mean()                                      
        grad_err = (grad_err_x + grad_err_y) / 2 

        return np.array(torch.tensor([torch.mean(abs_err), rmse, ratio_thresh, abs_err_01, abs_err_1, le90, grad_err], device='cpu'))
    

    @make_nograd_func
    def compute_values_rhf(self, ele_gt, ele_pred, ele_mask):
        ele_gt_masked = ele_gt[ele_mask]
        ele_pred_masked = ele_pred[ele_mask]
        abs_err = torch.abs(ele_gt_masked - ele_pred_masked)
        rmse = (ele_gt_masked - ele_pred_masked) ** 2
        rmse = torch.sqrt(rmse.mean())

        err_mask = abs_err > 0.5
        ratio_thresh = torch.mean(err_mask.float())

                  
        epsilon = 1e-6                               

                                   
        abs_err_01 = torch.mean((abs_err > 0.1).float())                               
        abs_err_1 = torch.mean((abs_err > 1.0).float())                                

                              
        le90 = torch.quantile(abs_err, 0.9)                                     

                                                                                 
        mask_gx = ele_mask[:, 1:] & ele_mask[:, :-1]
        mask_gy = ele_mask[1:, :] & ele_mask[:-1, :]

        grad_pred_x = ele_pred[:, 1:] - ele_pred[:, :-1]
        grad_pred_y = ele_pred[1:, :] - ele_pred[:-1, :]
        grad_gt_x = ele_gt[:, 1:] - ele_gt[:, :-1]
        grad_gt_y = ele_gt[1:, :] - ele_gt[:-1, :]

        grad_err_x = torch.abs(grad_pred_x - grad_gt_x)[mask_gx].mean() if mask_gx.any() else torch.tensor(0.0)
        grad_err_y = torch.abs(grad_pred_y - grad_gt_y)[mask_gy].mean() if mask_gy.any() else torch.tensor(0.0)
        grad_err = (grad_err_x + grad_err_y) / 2
    
        return np.array(torch.tensor([torch.mean(abs_err), rmse, ratio_thresh, abs_err_01, abs_err_1, le90, grad_err], device='cpu'))


    @make_nograd_func
    def compute(self, ele_pred, ele_gt, ele_mask):
                             
        mask_roi = torch.logical_and(ele_gt > -self.ele_range, ele_gt < self.ele_range)
                                                    
        ele_mask = torch.logical_and(mask_roi, ele_mask)
                                                    
                                                      
        ele_mask = ele_mask.bool()
                                              
        if ele_mask.sum() == 0:
            print("Skipping computation due to zero ele_mask")
            return

        self.count_all += 1
                                                                                     
        self.metric_all += self.compute_values_rhf(ele_gt.squeeze(), ele_pred.squeeze(), ele_mask.squeeze())

        if self.distance_wise:
            for i in range(self.num_intervals):
                try:
                    ele_gt_ = ele_gt[:, i * self.intervals:(i+1)*self.intervals, :]
                    ele_pred_ = ele_pred[:, i*self.intervals:(i+1)*self.intervals, :]
                    ele_mask_ = ele_mask[:, i*self.intervals:(i+1)*self.intervals, :]
                except:
                    ele_gt_ = ele_gt[:, i*self.intervals:, :]
                    ele_pred_ = ele_pred[:, i*self.intervals:, :]
                    ele_mask_ = ele_mask[:, i*self.intervals:, :]
                gt_valid = ele_gt_[ele_mask_]
                if len(gt_valid) > 0:
                    pred_valid = ele_pred_[ele_mask_]
                    values = self.compute_values(gt_valid, pred_valid)
                    self.metric_wise[i, :] += values
                    self.count_wise[i] += 1
