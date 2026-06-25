import torch
import numpy as np
import cv2, random
from typing import Tuple

def _apply_flip(imgs_left, intrinsic, voxel_uv_left, ele_gt, mask, down_scale=4):
    """
    Horizontally flip the image and adjust related parameters.
    
    Args:
        imgs_left: torch.Tensor, shape (C, H, W), normalized to [0, 1]
        intrinsic: torch.Tensor, shape (3, 3), camera intrinsic matrix
        voxel_uv_left: torch.Tensor, shape (N, 2), UV coordinates (long/int type) in feature map space
        ele_gt: torch.Tensor, shape (Z, X), elevation ground truth (float32)
        mask: torch.Tensor, shape (Z, X), valid region mask (int8)
        down_scale: int, downscale factor between image and feature map (voxel UVs are in feature map space)
    
    Returns:
        imgs_left_flipped, intrinsic_flipped, voxel_uv_flipped, ele_gt_flipped, mask_flipped
        (all torch.Tensor with same dtype as input)
    """
                                           
    _, height, width = imgs_left.shape
                                                                   
    feat_width = width // down_scale
    
                                          
    imgs_left_flipped = torch.flip(imgs_left, dims=[2])
    
                                                                 
    intrinsic_flipped = intrinsic.clone()
    intrinsic_flipped[0, 2] = width - intrinsic[0, 2]                         
    
                                                                        
    voxel_uv_flipped = voxel_uv_left.clone()
    voxel_uv_flipped[0] = feat_width - 1 - voxel_uv_left[0]
    
                                                                        
    ele_gt_flipped = torch.flip(ele_gt, dims=[-1])
    mask_flipped = torch.flip(mask, dims=[-1])
    
    return imgs_left_flipped, intrinsic_flipped, voxel_uv_flipped, ele_gt_flipped, mask_flipped
 
 
def apply_gaussian_noise_and_blur(imgs_left, noise_sigma=0.01, blur_kernel_size=5):
    """
    Apply Gaussian noise and blur to the image.
    
    Args:
        imgs_left: torch.Tensor, shape (C, H, W), normalized to [0, 1]
        noise_sigma: Standard deviation of Gaussian noise (default: 0.01)
        blur_kernel_size: Size of the Gaussian blur kernel (default: 5, must be odd)
    
    Returns:
        Image with Gaussian noise and blur applied (torch.Tensor, same shape and dtype)
    """
    
                          
    device = imgs_left.device
    dtype = imgs_left.dtype
    
                                     
    imgs_np = imgs_left.cpu().numpy()                    
    
                                         
    imgs_np = np.transpose(imgs_np, (1, 2, 0))
    
                                   
    imgs_np = imgs_np.astype(np.float32)
    
                        
    noise = np.random.normal(0, noise_sigma, imgs_np.shape)
    imgs_noisy = imgs_np + noise
    imgs_noisy = np.clip(imgs_noisy, 0, 1)
    
                                         
    imgs_blurred = np.zeros_like(imgs_noisy)
    for c in range(imgs_noisy.shape[2]):
        imgs_blurred[:, :, c] = cv2.GaussianBlur(imgs_noisy[:, :, c], 
                                                  (blur_kernel_size, blur_kernel_size), 0)
    
                               
    imgs_blurred = np.transpose(imgs_blurred, (2, 0, 1))
    
                                                                 
    imgs_blurred_tensor = torch.from_numpy(imgs_blurred).to(dtype=dtype, device=device)
    
    return imgs_blurred_tensor
 
 
def apply_gt_cutout(ele_gt: torch.Tensor,
                    mask: torch.Tensor,
                    num_patches: int = 4,
                    patch_size: int = 10) -> Tuple:
    """
    Randomly zeros out rectangular patches of the GT supervision mask.
    Forces the model to interpolate / generalise rather than memorise
    the exact LiDAR pattern. ele_gt values are left untouched so the
    patches can be reinstated easily during evaluation.
    Safe to use: does NOT require any update to intrinsics or voxel_uv.

    Args:
        ele_gt          (H, W) elevation ground-truth tensor
        mask            (H, W) supervision mask  (1 = valid)
        num_patches     number of rectangular patches to blank out
        patch_size      side length of each square patch (pixels in BEV grid)

    Returns:
        ele_gt          unchanged (H, W) tensor
        mask_out        (H, W) mask with patches set to 0
    """
    mask_out = mask.clone()
    H, W     = mask.shape

    for _ in range(num_patches):
                                                       
        ph = min(patch_size, H)
        pw = min(patch_size, W)
        r  = random.randint(0, H - ph)
        c  = random.randint(0, W - pw)
        mask_out[r : r + ph, c : c + pw] = 0

    return ele_gt, mask_out

import numpy as np


def npz_to_ply(npz_path, ply_path, points_key=None):
    """
    Convert a point cloud stored in an NPZ file to a PLY file.

    Parameters
    ----------
    npz_path : str
        Path to input .npz file.
    ply_path : str
        Path to output .ply file.
    points_key : str, optional
        Key containing the point cloud inside the npz file.
        If None, the first array will be used.
    """

              
    data = np.load(npz_path)

                  
    if points_key is None:
        points = data[list(data.keys())[0]]
    else:
        points = data[points_key]

                    
    if points.shape[1] < 3:
        raise ValueError("Point cloud must have at least XYZ coordinates")

    xyz = points[:, :3]

               
    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for p in xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

    print(f"Saved PLY file: {ply_path}")