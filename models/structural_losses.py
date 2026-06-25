"""
Structural and perceptual losses for road height estimation.
These losses emphasize local structure preservation (bumps, potholes)
over global pixel-wise accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleGradientLoss(nn.Module):
    """
    Multi-scale gradient loss: compute gradient matching at multiple resolutions.
    Lower scales capture coarse road geometry; higher scales capture fine details
    like speed bumps. This is more robust than single-scale gradient loss.
    """
    def __init__(self, scales=(1, 2, 4), weights=None):
        super().__init__()
        self.scales = scales
        self.weights = weights or [1.0 / s for s in scales]                                

    def _gradient(self, x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    def forward(self, pred, gt, mask):
        """
        pred, gt: (B, H, W)
        mask: (B, H, W) boolean
        """
        pred = pred.unsqueeze(1)                
        gt = gt.unsqueeze(1)
        mask = mask.unsqueeze(1).float()

        total_loss = 0.0
        for scale, w in zip(self.scales, self.weights):
            if scale > 1:
                p = F.avg_pool2d(pred, scale, scale)
                g = F.avg_pool2d(gt, scale, scale)
                m = F.avg_pool2d(mask, scale, scale)
                m = (m > 0.99).float()                                                 
            else:
                p, g, m = pred, gt, mask

            pdx, pdy = self._gradient(p)
            gdx, gdy = self._gradient(g)

                                           
            mx = m[:, :, :, 1:] * m[:, :, :, :-1]
            my = m[:, :, 1:, :] * m[:, :, :-1, :]

            loss_x = (torch.abs(pdx - gdx) * mx).sum() / mx.sum().clamp(min=1)
            loss_y = (torch.abs(pdy - gdy) * my).sum() / my.sum().clamp(min=1)

            total_loss += w * (loss_x + loss_y)

        return total_loss


class LocalStructureLoss(nn.Module):
    """
    SSIM-like local structural similarity for height maps.
    Compares local mean, variance, and covariance in sliding windows.
    This forces the network to match local height *patterns* (bump shape)
    rather than just per-pixel values.
    
    C1, C2 must be scaled to data range. For height in cm (range ~40cm):
      C1 = (0.01 * data_range)^2 = (0.01*40)^2 = 0.16
      C2 = (0.03 * data_range)^2 = (0.03*40)^2 = 1.44
    """
    def __init__(self, window_size=7, data_range=40.0):
        super().__init__()
        self.C1 = (0.01 * data_range) ** 2                       
        self.C2 = (0.03 * data_range) ** 2                       
        self.window_size = window_size
        self.pad = window_size // 2
                        
        self.register_buffer(
            'window',
            torch.ones(1, 1, window_size, window_size) / (window_size * window_size)
        )

    def forward(self, pred, gt, mask):
        """
        pred, gt: (B, H, W), in cm
        mask: (B, H, W) boolean
        """
        pred = pred.unsqueeze(1)                
        gt = gt.unsqueeze(1)
        mask_f = mask.unsqueeze(1).float()

        w = self.window.to(pred.device)
        eps = 1e-8
                                                            
        mu_p = F.conv2d(pred * mask_f, w, padding=self.pad)
        mu_g = F.conv2d(gt * mask_f, w, padding=self.pad)
        count = F.conv2d(mask_f, w, padding=self.pad).clamp(min=1e-3)
        mu_p = mu_p / count
        mu_g = mu_g / count

                                        
        sigma_pp = F.conv2d((pred * mask_f) ** 2, w, padding=self.pad) / count - mu_p ** 2
        sigma_gg = F.conv2d((gt * mask_f) ** 2, w, padding=self.pad) / count - mu_g ** 2
        sigma_pg = F.conv2d(pred * gt * mask_f, w, padding=self.pad) / count - mu_p * mu_g

                                       
        sigma_pp = sigma_pp.clamp(min=0)
        sigma_gg = sigma_gg.clamp(min=0)

        ssim = ((2 * mu_p * mu_g + self.C1) * (2 * sigma_pg + self.C2)) /\
               ((mu_p**2 + mu_g**2 + self.C1) * (sigma_pp + sigma_gg + self.C2)) + eps

                                                        
        valid_mask = (count > self.window_size * self.window_size * 0.5).float()
        loss = (1.0 - ssim) * valid_mask

        return loss.sum() / valid_mask.sum().clamp(min=1)


class EdgeAwareSmoothnessLoss(nn.Module):
    """
    Edge-aware smoothness: encourages flat predictions in smooth GT regions
    while allowing sharp transitions where GT has edges.
    This suppresses noise in flat road areas without blurring speed bumps.
    """
    def __init__(self, edge_scale=1.0):
        super().__init__()
        self.edge_scale = edge_scale                                           

    def forward(self, pred, gt, mask):
        """
        pred, gt: (B, H, W)
        mask: (B, H, W) boolean
        """
        pred = pred.unsqueeze(1)
        gt = gt.unsqueeze(1)
        mask_f = mask.unsqueeze(1).float()

                                           
        gt_dx = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
        gt_dy = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])

                                             
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

                                                                            
        weight_x = torch.exp(-gt_dx / self.edge_scale)
        weight_y = torch.exp(-gt_dy / self.edge_scale)

                              
        mx = mask_f[:, :, :, 1:] * mask_f[:, :, :, :-1]
        my = mask_f[:, :, 1:, :] * mask_f[:, :, :-1, :]

        loss_x = (pred_dx * weight_x * mx).sum() / mx.sum().clamp(min=1)
        loss_y = (pred_dy * weight_y * my).sum() / my.sum().clamp(min=1)

        return loss_x + loss_y

class L2Loss(nn.Module):
    """
    L2 Loss (Squared Euclidean Distance) without square root
    
    Computes the sum of squared differences between predictions and targets.
    L2 Loss = sum((y_pred - y_true)^2)
    
    Unlike MSE which divides by N, this returns the raw sum of squared differences.
    
    Args:
        reduction (str): Specifies the reduction to apply to the output.
            'mean': the mean of the output is taken (equivalent to MSE)
            'sum': the output will be summed
            'none': no reduction will be applied
            Default: 'sum'
    """
    
    def __init__(self, reduction='sum'):
        super(L2Loss, self).__init__()
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
        
        self.reduction = reduction
    
    def forward(self, input, target):
        """
        Args:
            input (Tensor): Predictions from the model
            target (Tensor): Ground truth targets
        
        Returns:
            Tensor: L2 loss value (sum of squared differences)
        """
                                     
        squared_diff = (input - target) ** 2
        
                         
        if self.reduction == 'mean':
            return squared_diff.mean()
        elif self.reduction == 'sum':
            return squared_diff.sum()
        else:          
            return squared_diff
        

class HeightDistributionLoss(nn.Module):
    """
    Penalizes mismatch in the histogram/distribution of predicted vs GT heights.
    Uses soft histogram matching to ensure the network produces the right
    *range* of heights (important for bumps which are rare height values).
    """
    def __init__(self, n_bins=64, ele_range=20.0):
        super().__init__()
        self.n_bins = n_bins
        self.ele_range = ele_range         
        bin_edges = torch.linspace(-ele_range, ele_range, n_bins + 1)
        self.register_buffer('bin_centers', (bin_edges[:-1] + bin_edges[1:]) / 2)
        self.sigma = (2 * ele_range) / n_bins             

    def soft_histogram(self, x, mask):
        """x: (B, H, W), mask: (B, H, W)"""
        x_flat = x[mask]        
        if x_flat.numel() == 0:
            return torch.zeros(self.n_bins, device=x.device)
        diffs = x_flat.unsqueeze(-1) - self.bin_centers.to(x.device).unsqueeze(0)             
        weights = torch.exp(-0.5 * (diffs / self.sigma) ** 2)
        hist = weights.sum(dim=0)
        return hist / hist.sum().clamp(min=1e-8)

    def forward(self, pred, gt, mask):
        mask = mask.bool()
        hist_pred = self.soft_histogram(pred, mask)
        hist_gt = self.soft_histogram(gt, mask)
                                 
        eps = 1e-8
        kl_pq = (hist_gt * torch.log((hist_gt + eps) / (hist_pred + eps))).sum()
        kl_qp = (hist_pred * torch.log((hist_pred + eps) / (hist_gt + eps))).sum()
        return 0.5 * (kl_pq + kl_qp)


class CompositeLoss(nn.Module):
    """
    Combines L1/MSE with structural losses for road height estimation.
    
    Default weights tuned for cm-scale height maps:
    - pixel_loss: primary supervision
    - gradient_loss: captures slopes/edges of bumps
    - structure_loss: local pattern matching (SSIM-like)
    - normal_loss: surface orientation consistency
    - smoothness_loss: noise suppression in flat areas
    """
    def __init__(self, ele_range, hori_centers=None, normalize=False,
                 pixel_type='L1',
                 w_pixel=1.0,
                 w_gradient=0.5,
                 w_structure=0.2,
                 w_normal=0.1,
                 w_smoothness=0.05):
        super().__init__()
        self.ele_range = ele_range * 100         
        self.normalize = normalize
        self.w_pixel = w_pixel
        self.w_gradient = w_gradient
        self.w_structure = w_structure
        self.w_normal = w_normal
        self.w_smoothness = w_smoothness
        print(f"Initialized CompositeLoss with weights: pixel={w_pixel}, gradient={w_gradient}, structure={w_structure}, normal={w_normal}, smoothness={w_smoothness}")
        if pixel_type == 'L1':
            self.pixel_loss = nn.L1Loss(reduction='mean')
        elif pixel_type == 'MSE':
            self.pixel_loss = nn.MSELoss(reduction='mean')
        else:
            self.pixel_loss = L2Loss(reduction='mean')                               
        self.gradient_loss = MultiScaleGradientLoss(scales=(1, 2, 4))
        self.structure_loss = LocalStructureLoss(window_size=7, data_range=self.ele_range * 2)
        self.smoothness_loss = EdgeAwareSmoothnessLoss(edge_scale=1.0)                

        self.hori_centers = hori_centers
        self.normal_loss_fn = None
        if hori_centers is not None:
            from utils.normals import normal_loss
            self.normal_loss_fn = normal_loss

    def forward(self, ele_pred, ele_gt, ele_mask):
        """
        ele_pred: (B, H, W) or (B, 1, H, W)
        ele_gt: (B, H, W)
        ele_mask: (B, H, W)
        """
        if ele_pred.dim() == 4:
            ele_pred = ele_pred.squeeze(1)

        roi_mask = (ele_gt > -self.ele_range) & (ele_gt < self.ele_range)
        mask = roi_mask & ele_mask.bool()

                                                                              
        if mask.sum() == 0 and ele_mask.bool().sum() > 0:
            mask = ele_mask.bool()

                                                        
        if mask.sum() == 0:
            loss_pixel = torch.tensor(0.0, device=ele_pred.device)
        else:
            loss_pixel = self.pixel_loss(ele_pred[mask], ele_gt[mask])

                                   
        loss_grad = self.gradient_loss(ele_pred, ele_gt, mask) if self.w_gradient > 0 else torch.tensor(0.0, device=ele_pred.device)

                               
        loss_struct = self.structure_loss(ele_pred, ele_gt, mask) if self.w_structure > 0 else torch.tensor(0.0, device=ele_pred.device)

                         
        loss_smooth = self.smoothness_loss(ele_pred, ele_gt, mask) if self.w_smoothness > 0 else torch.tensor(0.0, device=ele_pred.device)

        total = (self.w_pixel * loss_pixel +
                 self.w_gradient * loss_grad +
                 self.w_structure * loss_struct +
                 self.w_smoothness * loss_smooth)

                                                
        loss_normal = torch.tensor(0.0, device=ele_pred.device)
        if self.normal_loss_fn is not None and self.hori_centers is not None:
            loss_normal = self.normal_loss_fn(
                ele_pred, ele_gt,
                self.hori_centers, mask
            )
            total += self.w_normal * loss_normal
                                  
        print(f"nan check - pixel: {loss_pixel.isnan().item()}, grad: {loss_grad.isnan().item()}, struct: {loss_struct.isnan().item()}, smooth: {loss_smooth.isnan().item()}, normal: {loss_normal.isnan().item()}")
                                                 
        self._last_components = {
            'pixel': loss_pixel.detach().item(),
            'gradient': loss_grad.detach().item() if isinstance(loss_grad, torch.Tensor) else loss_grad,
            'structure': loss_struct.detach().item() if isinstance(loss_struct, torch.Tensor) else loss_struct,
            'smoothness': loss_smooth.detach().item() if isinstance(loss_smooth, torch.Tensor) else loss_smooth,
            'normal': loss_normal.detach().item(),
        }

        return total
