import torch.nn.functional as F
import torch
from torch import nn
from typing import Union, Tuple, Optional, Callable
import math
import utils3d
from utils.normals import normal_loss

class MyLoss(nn.Module):
    def __init__(self, ele_range, voxel_ele_res, cla_res=1):
        super(MyLoss, self).__init__()
        self.ele_range = ele_range*100         
        if (self.ele_range*20) % (cla_res*10) != 0:
            print('The class interval is improper')
            exit()
        self.cla_res = cla_res          
        self.voxel_ele_res = voxel_ele_res*100         
        self.num_voxels_ele = int(self.ele_range*2 / self.voxel_ele_res)

        self.num_classes = int(2*self.ele_range/cla_res)
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def label2class(self, ele_gt):
                      
        if ele_gt.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=ele_gt.device)
        class_label = torch.floor((ele_gt + self.ele_range) / self.cla_res).type(torch.long)
        class_label = self.num_classes - class_label - 1

        return class_label

    def forward(self, ele_pred, ele_gt, ele_mask):
                                                           
                             
        ele_mask_roi = torch.logical_and(ele_gt > -self.ele_range, ele_gt < self.ele_range)
        ele_mask = torch.logical_and(ele_mask_roi, ele_mask)

        ele_pred = ele_pred.permute(0, 2, 3, 1)
        ele_pred = ele_pred[ele_mask, :]
        ele_gt = ele_gt[ele_mask]

                                             
        if ele_gt.numel() == 0:
            return torch.tensor(0.0, device=ele_pred.device, requires_grad=True)

                                                         
        class_ele = self.label2class(ele_gt)
        loss_ele = self.loss_func(ele_pred, class_ele)

        return loss_ele
      
class LossReg(nn.Module):
    def __init__(self, ele_range, normalize=False, type_of_loss = 'L1'):
        super(LossReg, self).__init__()
        self.ele_range = ele_range*100
        if type_of_loss == 'L1':
            self.loss_func = nn.L1Loss(reduction='mean')
            self.loss_type = "L1"
        elif type_of_loss == 'MSE':
            self.loss_func = nn.MSELoss(reduction='mean')
            self.loss_type = "MSE"
        elif type_of_loss == "lpips":
            self.loss_type = "lpips"
            self.loss_func = lpips.LPIPS(net = "vgg")
            print("using perceptual loss")
        self.normalize = normalize

    def forward(self, ele_pred, ele_gt, ele_mask):
                             
                             
        print("ele_gt max and min", ele_gt.max(), ele_gt.min(), ele_gt.mean())
        ele_mask_roi = torch.logical_and(ele_gt > -self.ele_range, ele_gt < self.ele_range)
                                                     
                                             
        ele_mask = torch.logical_and(ele_mask_roi, ele_mask)
        ele_mask = ele_mask.bool()
        print("ele_mask:" , ele_mask.shape)
        ele_pred_masked = ele_pred[ele_mask]
                                    
                                                         
        total_cell = ele_gt.shape[-1] * ele_gt.shape[-2]
        ele_gt_masked = ele_gt[ele_mask]
        gt_min = - self.ele_range
        gt_max = self.ele_range
        if self.normalize:
            print("normalized")
            gt_scaled = (2 * ele_gt_masked - (gt_max + gt_min)) / (gt_max - gt_min)
                                                                                       

            assert(gt_scaled.shape == ele_gt.shape)
            loss = self.loss_func(ele_pred_masked, gt_scaled)
        else:
            print("masked prediction", ele_pred_masked.shape)
            print("mask", ele_gt_masked.shape)

            loss = self.loss_func(ele_pred_masked, ele_gt_masked)

        return loss
    
class GradientLoss(nn.Module):
    """
    Computes L1 loss between spatial gradients of prediction and ground truth.
    """

    def __init__(self):
        super().__init__()
    @staticmethod
    def gradient_x(img):
        return img[:, :, :, 1:] - img[:, :, :, :-1]

    @staticmethod
    def gradient_y(img):
        return img[:, :, 1:, :] - img[:, :, :-1, :]

    def forward(self, pred, gt, mask):
        """
        pred: (B, 1, H, W)
        gt:   (B, 1, H, W)
        mask: (B, 1, H, W) boolean
        """
        print(pred.shape)
        print(gt.shape)

        pred_dx = self.gradient_x(pred)
        pred_dy = self.gradient_y(pred)

        gt_dx = self.gradient_x(gt)
        gt_dy = self.gradient_y(gt)

        print(pred_dx.shape)
        print(gt_dx.shape)
                                                                
        mask_dx = mask[:, :, :, 1:] & mask[:, :, :, :-1]
        mask_dy = mask[:, :, 1:, :] & mask[:, :, :-1, :]

        loss_x = torch.abs(pred_dx - gt_dx)[mask_dx].mean()
        loss_y = torch.abs(pred_dy - gt_dy)[mask_dy].mean()

        return loss_x + loss_y

class HeteroscedasticNLLLoss(nn.Module):
    """
    Gaussian negative log-likelihood loss with learned per-pixel variance.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        """
        pred: (B, 2, H, W)
        gt:   (B, 1, H, W)
        mask: (B, 1, H, W) boolean
        """
        mean = pred[:, 0:1]
        var  = pred[:, 1:2]

        var = F.softplus(var) + self.eps

        nll = (gt - mean) ** 2 / (2.0 * var) + 0.5 * torch.log(var)

        return nll[mask].mean()
    
    
class LossReg2(nn.Module):                            
    def __init__(self, ele_range, gradient_weight=0.01, normalize=False, type_of_loss = "L1"):
        super(LossReg2, self).__init__()
        self.normalize = normalize 
        self.gradientloss = GradientLoss()
        self.nll = HeteroscedasticNLLLoss()
        self.ele_range = ele_range*100
        self.gradient_weight = gradient_weight
        if type_of_loss == 'L1':
            self.l1loss = nn.L1Loss(reduction='mean')
        elif type_of_loss == 'MSE':
            self.l1loss = nn.MSELoss(reduction='mean')
            
    def forward(self, ele_pred, ele_gt, ele_mask):
                                                   
                             
        gt_min = - self.ele_range
        gt_max = self.ele_range
        if self.normalize:
            print("normalized")
            pred_scaled = (ele_pred * (gt_max - gt_min) / 2) + ((gt_max + gt_min) / 2)

            assert(pred_scaled.shape == ele_pred.shape)
            ele_pred = pred_scaled
                                
        roi_mask = torch.logical_and((ele_gt > -self.ele_range),(ele_gt < self.ele_range))
        mask = torch.logical_and(roi_mask,ele_mask)
        mask = mask.unsqueeze(1)                

        ele_gt = ele_gt.unsqueeze(1)
                                         
        ele_pred = ele_pred.unsqueeze(1)
                                                    
        loss_grad = self.gradientloss(ele_pred, ele_gt, mask)

        ele_pred = ele_pred[mask]
        ele_gt = ele_gt[mask]
        l1loss = self.l1loss(ele_pred, ele_gt)

        return l1loss + self.gradient_weight * loss_grad                      

class normalloss(nn.Module):
    def __init__(self, ele_range, hori_centers):
        super().__init__()
        self.ele_range = ele_range*100
        self.hori_centers = hori_centers
    
    def forward(self, ele_pred, ele_gt, ele_mask):
        roi_mask = torch.logical_and((ele_gt > -self.ele_range),(ele_gt < self.ele_range))
        mask = torch.logical_and(roi_mask,ele_mask)
     
        loss = normal_loss(ele_pred, ele_gt, self.hori_centers, mask)

        return loss

class MSE_normal_loss(nn.Module):
    def __init__(self, ele_range, hori_centers, normalize = False):
        super().__init__()
        self.ele_range = ele_range
        self.hori_centers = hori_centers
        self.normal_loss = normalloss(ele_range, hori_centers)
        self.MSE_loss = LossReg(ele_range, normalize, type_of_loss='MSE')

    def forward(self, ele_pred, ele_gt, ele_mask):
        MSE_loss = self.MSE_loss(ele_pred, ele_gt, ele_mask)
        normal_loss = self.normal_loss(ele_pred, ele_gt, ele_mask)

        print("loss information \n")
        print("MSE_loss: ", MSE_loss)
        print("normal_loss:", normal_loss)

        return MSE_loss + 0.1 * normal_loss


class LossReg3(nn.Module):
    def __init__(self, ele_range, gradient_weight):
        super().__init__()
        self.ele_range = ele_range*100
        self.gradient_weight = gradient_weight
        self.l1loss = nn.L1Loss(reduction='mean')

    def forward(self, ele_pred, ele_gt, ele_mask):
        roi_mask = torch.logical_and((ele_gt > -self.ele_range),(ele_gt < self.ele_range))
        mask = torch.logical_and(roi_mask,ele_mask)
        mask = mask.unsqueeze(1)                
     
        ele_gt = ele_gt[mask]
        ele_pred = ele_pred[mask]
        loss = self.l1loss(ele_pred, ele_gt)

        return loss
    

class affine_invariant_global_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_points, gt_points, align_res, beta, trunc, sparsity_aware):
        mask = torch.isfinite(gt_points).all(dim=-1)
        gt_points = torch.where(mask[..., None], gt_points, 1)

               
        pred_points_lr, gt_points_lr, lr_mask = utils3d.pt.masked_nearest_resize(pred_points, gt_points, mask=mask, size=(align_res, align_res))
        scale, shift = align_points_scale_z_shift(pred_points_lr.flatten(-3, -2), gt_points_lr.flatten(-3, -2), lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), trunc=trunc)
        valid = scale > 0
        scale, shift = torch.where(valid, scale, 0), torch.where(valid[..., None], shift, 0)

        pred_points = scale[..., None, None, None] * pred_points + shift[..., None, None, :]

                      
        weight = (valid[..., None, None] & mask).float() / gt_points[..., 2].clamp_min(1e-5)
        weight = weight.clamp_max(10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))                                                            
        loss = _smooth((pred_points - gt_points).abs() * weight[..., None], beta=beta).mean(dim=(-3, -2, -1))

        if sparsity_aware:
                                                                                                      
            sparsity = mask.float().mean(dim=(-2, -1)) / lr_mask.float().mean(dim=(-2, -1))
            loss = loss / (sparsity + 1e-7)

        err = (pred_points.detach() - gt_points).norm(dim=-1) / gt_points[..., 2]

                                  
        misc = {
            'truncated_error': weighted_mean(err.clamp_max(1.0), mask).item(),
            'delta': weighted_mean((err < 1).float(), mask).item()
        }

        return loss, misc, scale.detach()


def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)
    
def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)
    
def align_points_scale_z_shift(points_src: torch.Tensor, points_tgt: torch.Tensor, weight: Optional[torch.Tensor], trunc: Optional[Union[float, torch.Tensor]] = None):
    """
    Align `points_src` to `points_tgt` with respect to a shared xyz scale and z shift. 
    It is similar to `align_affine` but scale and shift are applied to different dimensions.

    ### Parameters:
    - `points_src: torch.Tensor` of shape (..., N, 3)
    - `points_tgt: torch.Tensor` of shape (..., N, 3)
    - `weights: torch.Tensor` of shape (..., N)

    ### Returns:
    - `scale: torch.Tensor` of shape (...).
    - `shift: torch.Tensor` of shape (..., 3). x and y shifts are zeros.
    """
    dtype, device = points_src.dtype, points_src.device

                                             
    batch_shape, n = points_src.shape[:-2], points_src.shape[-2]
    batch_size = math.prod(batch_shape)
    points_src, points_tgt, weight = points_src.reshape(batch_size, n, 3), points_tgt.reshape(batch_size, n, 3), weight.reshape(batch_size, n)

                  
    anchor_where_batch, anchor_where_n = torch.where(weight > 0)
    with torch.no_grad():
        zeros = torch.zeros(anchor_where_batch.shape[0], device=device, dtype=dtype)
        points_src_anchor = torch.stack([zeros, zeros, points_src[anchor_where_batch, anchor_where_n, 2]], dim=-1)                    
        points_tgt_anchor = torch.stack([zeros, zeros, points_tgt[anchor_where_batch, anchor_where_n, 2]], dim=-1)                    

        points_src_anchored = points_src[anchor_where_batch, :, :] - points_src_anchor[..., None, :]                     
        points_tgt_anchored = points_tgt[anchor_where_batch, :, :] - points_tgt_anchor[..., None, :]                     
        weight_anchored = weight[anchor_where_batch, :, None].expand(-1, -1, 3)                                          

                                                       
        MAX_ELEMENTS = 2 ** 20
        scale, loss, index = split_batch_fwd(align, MAX_ELEMENTS // n, points_src_anchored.flatten(-2), points_tgt_anchored.flatten(-2), weight_anchored.flatten(-2), trunc)               

        loss, index_anchor = scatter_min(size=batch_size, dim=0, index=anchor_where_batch, src=loss)                   

                                                     
    index_2 = index[index_anchor]                                                      
    index_1 = anchor_where_n[index_anchor] * 3 + index_2 % 3                           

    zeros = torch.zeros((batch_size, n), device=device, dtype=dtype)
    points_tgt_00z, points_src_00z = torch.stack([zeros, zeros, points_tgt[..., 2]], dim=-1), torch.stack([zeros, zeros, points_src[..., 2]], dim=-1)
    tgt_1, src_1 = torch.gather(points_tgt_00z.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1), torch.gather(points_src_00z.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1)
    tgt_2, src_2 = torch.gather(points_tgt.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1), torch.gather(points_src.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1)

    scale = (tgt_2 - tgt_1) / torch.where(src_2 != src_1, src_2 - src_1, 1.0)
    shift = torch.gather(points_tgt_00z, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2) - scale[..., None] * torch.gather(points_src_00z, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2)
    scale, shift = scale.reshape(batch_shape), shift.reshape(*batch_shape, 3)

    return scale, shift


def scatter_min(size: int, dim: int, index: torch.LongTensor, src: torch.Tensor):
    "Scatter the minimum value along the given dimension of `input` into `src` at the indices specified in `index`."
    shape = src.shape[:dim] + (size,) + src.shape[dim + 1:]
    minimum = torch.full(shape, float('inf'), dtype=src.dtype, device=src.device).scatter_reduce(dim=dim, index=index, src=src, reduce='amin', include_self=False)
    minimum_where = torch.where(src == torch.gather(minimum, dim=dim, index=index))
    indices = torch.full(shape, -1, dtype=torch.long, device=src.device)
    indices[(*minimum_where[:dim], index[minimum_where], *minimum_where[dim + 1:])] = minimum_where[dim]
    return torch.return_types.min((minimum, indices))


def split_batch_fwd(fn: Callable, chunk_size: int, *args, **kwargs):
    batch_size = next(x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)).shape[0]
    n_chunks = batch_size // chunk_size + (batch_size % chunk_size > 0)
    splited_args = tuple(arg.split(chunk_size, dim=0) if isinstance(arg, torch.Tensor) else [arg] * n_chunks for arg in args)
    splited_kwargs = {k: [v.split(chunk_size, dim=0) if isinstance(v, torch.Tensor) else [v] * n_chunks] for k, v in kwargs.items()}
    results = []
    for i in range(n_chunks):
        chunk_args = tuple(arg[i] for arg in splited_args)
        chunk_kwargs = {k: v[i] for k, v in splited_kwargs.items()}
        results.append(fn(*chunk_args, **chunk_kwargs))

    if isinstance(results[0], tuple):
        return tuple(torch.cat(r, dim=0) for r in zip(*results))
    else:
        return torch.cat(results, dim=0)

def align(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, trunc: Optional[Union[float, torch.Tensor]] = None, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """
    If trunc is None, solve `min sum_i w_i * |a * x_i - y_i|`, otherwise solve `min sum_i min(trunc, w_i * |a * x_i - y_i|)`.
    
    w_i must be >= 0.

    ### Parameters:
    - `x`: tensor of shape (..., n)
    - `y`: tensor of shape (..., n)
    - `w`: tensor of shape (..., n)
    - `trunc`: optional, float or tensor of shape (..., n) or None

    ### Returns:
    - `a`: tensor of shape (...), differentiable
    - `loss`: tensor of shape (...), value of loss function at `a`, detached
    - `index`: tensor of shape (...), where a = y[idx] / x[idx]
    """
    if trunc is None:
        x, y, w = torch.broadcast_tensors(x, y, w)
        sign = torch.sign(x)
        x, y = x * sign, y * sign
        y_div_x = y / x.clamp_min(eps)
        y_div_x, argsort = y_div_x.sort(dim=-1)

        wx = torch.gather(x * w, dim=-1, index=argsort)
        derivatives = 2 * wx.cumsum(dim=-1) - wx.sum(dim=-1, keepdim=True)
        search = torch.searchsorted(derivatives, torch.zeros_like(derivatives[..., :1]), side='left').clamp_max(derivatives.shape[-1] - 1)

        a = y_div_x.gather(dim=-1, index=search).squeeze(-1)
        index = argsort.gather(dim=-1, index=search).squeeze(-1)
        loss = (w * (a[..., None] * x - y).abs()).sum(dim=-1)
        
    else:
                                                   
        x, y, w = torch.broadcast_tensors(x, y, w)
        batch_shape = x.shape[:-1]
        batch_size = math.prod(batch_shape)
        x, y, w = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), w.reshape(-1, w.shape[-1])

        sign = torch.sign(x)
        x, y = x * sign, y * sign
        wx, wy = w * x, w * y
        xyw = torch.stack([x, y, w], dim=-1)                                      

        y_div_x = A = y / x.clamp_min(eps)
        B = (wy - trunc) / wx.clamp_min(eps)
        C = (wy + trunc) / wx.clamp_min(eps)
        with torch.no_grad():
                                                          
            A, A_argsort = A.sort(dim=-1)
            Q_A = torch.cumsum(torch.gather(wx, dim=-1, index=A_argsort), dim=-1)
            A, Q_A = _pad_inf(A), _pad_cumsum(Q_A)                                                                                 

            B, B_argsort = B.sort(dim=-1)
            Q_B = torch.cumsum(torch.gather(wx, dim=-1, index=B_argsort), dim=-1)
            B, Q_B = _pad_inf(B), _pad_cumsum(Q_B)

            C, C_argsort = C.sort(dim=-1)
            Q_C = torch.cumsum(torch.gather(wx, dim=-1, index=C_argsort), dim=-1)
            C, Q_C = _pad_inf(C), _pad_cumsum(Q_C)
            
                                                     
            j_A = torch.searchsorted(A, y_div_x, side='left').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='left').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='left').sub_(1)
            left_derivative = 2 * torch.gather(Q_A, dim=-1, index=j_A) - torch.gather(Q_B, dim=-1, index=j_B) - torch.gather(Q_C, dim=-1, index=j_C)
            j_A = torch.searchsorted(A, y_div_x, side='right').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='right').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='right').sub_(1)
            right_derivative = 2 * torch.gather(Q_A, dim=-1, index=j_A) - torch.gather(Q_B, dim=-1, index=j_B) - torch.gather(Q_C, dim=-1, index=j_C)

                          
            is_extrema = (left_derivative < 0) & (right_derivative >= 0)
            is_extrema[..., 0] |= ~is_extrema.any(dim=-1)                                                                                         
            where_extrema_batch, where_extrema_index = torch.where(is_extrema)          

                                                  
            extrema_a = y_div_x[where_extrema_batch, where_extrema_index]                               
            MAX_ELEMENTS = 4096 ** 2                                                                                      
            SPLIT_SIZE = MAX_ELEMENTS // x.shape[-1]
            extrema_value = torch.cat([
                _compute_residual(extrema_a_split[:, None], xyw[extrema_i_split, :, :], trunc)
                for extrema_a_split, extrema_i_split in zip(extrema_a.split(SPLIT_SIZE), where_extrema_batch.split(SPLIT_SIZE))
            ])                          
            
                                                     
            minima, indices = scatter_min(size=batch_size, dim=0, index=where_extrema_batch, src=extrema_value)                       
            index = where_extrema_index[indices]

        a = torch.gather(y, dim=-1, index=index[..., None]) / torch.gather(x, dim=-1, index=index[..., None]).clamp_min(eps)
        a = a.reshape(batch_shape)
        loss = minima.reshape(batch_shape)
        index = index.reshape(batch_shape)

    return a, loss, index

def _pad_inf(x_: torch.Tensor):
    return torch.cat([torch.full_like(x_[..., :1], -torch.inf), x_, torch.full_like(x_[..., :1], torch.inf)], dim=-1)


def _pad_cumsum(cumsum: torch.Tensor):
    return torch.cat([torch.zeros_like(cumsum[..., :1]), cumsum, cumsum[..., -1:]], dim=-1)

def _compute_residual(a: torch.Tensor, xyw: torch.Tensor, trunc: float):
    return a.mul(xyw[..., 0]).sub_(xyw[..., 1]).abs_().mul_(xyw[..., 2]).clamp_max_(trunc).sum(dim=-1)


class CustomMSELoss(nn.Module):
    """
    Custom MSE Loss that computes:
    - Sum of squared errors for each sample
    - Mean across the batch
    
    This is useful when you want to weight each sample equally
    regardless of how many valid pixels each sample has.
    """
    
    def __init__(self, reduction='mean'):
        super(CustomMSELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, target, mask, total_valid_pixel):
        """
        Args:
            pred: [B, ...] - Predicted values
            target: [B, ...] - Target values
        
        Returns:
            loss: scalar - Loss value
        """
                               
        print("target", target.shape)
        print("pred", pred.shape)
        squared_error = (pred - target) ** 2
        
                                                      
        batch_size = squared_error.shape[0]
        squared_error_flat = squared_error.reshape(batch_size, -1)
        
                             
        per_sample_loss = squared_error_flat.sum(dim=1)       
        n_possible = total_valid_pixel
        if mask is not None:
            mask_flat = mask.reshape(batch_size, -1).float()                
            n_valid = mask_flat.sum(dim=1)                               

            weight = n_valid / n_possible                                

            per_sample_loss = (squared_error_flat * mask_flat).sum(dim=1)       
        else:
            weight = 1.0

        weighted_loss = weight * per_sample_loss                         

        possible_valid_cells = squared_error.shape[-1] * squared_error.shape[-2]
        print("possible valid:", possible_valid_cells)
        
        if self.reduction == 'mean':
                               
            return weighted_loss.mean()
        elif self.reduction == 'sum':
                              
            return weighted_loss.sum()
        else:
                                      
            return weighted_loss

class Custom_L1loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(Custom_L1loss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, target):
        l1_error = pred - target

        B = l1_error.shape[0]
        l1_error.reshape(B, -1)
        
        per_sample_error = l1_error.sum(dim=1)

        if self.reduction == 'mean':
            return per_sample_error.mean()
        elif self.reduction == 'sum':
            return per_sample_error.sum()
        else:
            return per_sample_error