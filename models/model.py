from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from typing import List, Optional
from models.ele_head import *
import math
from .efficientnet import efficientnet_feature
from utils.experiment import save_feature_map
from .patch2feature import _make_scratch, _make_fusion_block, patch2feature, easy_transition_layer, make_pca, DinoUpsampler
import warnings
import cv2
from contextlib import nullcontext
from sklearn.decomposition import PCA
import numpy as np
import os

import sys
sys.path.append('/home/f9ql00v/depth-anything3/Depth-Anything-3-main/src')


from depth_anything_3.api import DepthAnything3

def print_types(obj, indent=0):
    prefix = "  " * indent

    if isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} (len={len(obj)})")
        for i, item in enumerate(obj):
           print(f"{prefix}  [{i}]:")
           print_types(item, indent + 2)
    else:
       print(f"{prefix}{type(obj).__name__}")

def visualize_encoder_pca(features, save_path, patch_size=14, img_hw=None, layer_idx=-1, batch_idx=0):
    """
    Visualize encoder features via PCA -> RGB image.

    Args:
        features: encoder output. Supported shapes:
            - Tensor [B, C, H, W]                       (e.g. EfficientNet)
            - Tensor [B, N, C] with N = ph*pw (+ CLS)   (e.g. DINO single layer)
            - List/tuple of [B, N, C] tensors           (e.g. DINO intermediate layers)
        save_path:  where to write the PNG.
        patch_size: ViT patch size (only used for token-shaped features).
        img_hw:     (H, W) of the input image, required for token-shaped features
                    so we can recover the spatial grid (ph = H // patch_size).
        layer_idx:  which layer to visualize when `features` is a list.
        batch_idx:  which sample of the batch to visualize.
    """
    if isinstance(features, (list, tuple)):
        feat = features[layer_idx]
    else:
        feat = features

    feat = feat.detach().float().cpu()

    if feat.dim() == 4:
        f = feat[batch_idx].permute(1, 2, 0)
    elif feat.dim() == 3:
        assert img_hw is not None, "img_hw=(H, W) required for token-shaped features"
        H, W = img_hw
        ph, pw = H // patch_size, W // patch_size
        f = feat[batch_idx]
        if f.shape[0] == ph * pw + 1:
            f = f[1:]
        elif f.shape[0] != ph * pw:
            raise ValueError(
                f"Token count {f.shape[0]} does not match ph*pw={ph*pw} (+CLS)"
            )
        f = f.reshape(ph, pw, -1)
    else:
        raise ValueError(f"Unsupported feature shape: {tuple(feat.shape)}")

    H, W, C = f.shape
    flat = f.reshape(-1, C).numpy()

    pca = PCA(n_components=3)
    rgb = pca.fit_transform(flat).reshape(H, W, 3)
    mn = rgb.min(axis=(0, 1), keepdims=True)
    mx = rgb.max(axis=(0, 1), keepdims=True)
    rgb = (rgb - mn) / (mx - mn + 1e-8)
    rgb = (rgb * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, rgb[:, :, ::-1])
    return rgb


class Elevation(nn.Module):
    def __init__(self, stereo,  num_grids, ele_range, cla_res, regression=False, backbone = 'efficientnet', normalize=False, pred_dim =256, train_encoder=False):
        super(Elevation, self).__init__()
        self.stereo = stereo
        self.num_grids_x, self.num_grids_y, self.num_grids_z = num_grids
        self.ele_range = ele_range             
        self.regression = regression
        self.backbone = backbone
        self.train_encoder = train_encoder

        self.cla_res = cla_res
        self.num_classes = int(2 * self.ele_range*100 / self.cla_res)                                                       
        ele_values = -torch.arange(self.num_classes, dtype=torch.float32, device='cuda')*self.cla_res + self.ele_range*100 - self.cla_res/2                                                 
        self.ele_values = ele_values.reshape(1, self.num_classes, 1, 1)
        
        self.patch2feat = True
                                                                                                      
        self.upsampler_kind = 'patch2feature'
                                                           
        self.patchsize = int(14)
        self._pca_viz_done = False
        if 'DepthAnything3' in backbone :
            model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
                                                 
            encoder = model.model.backbone
                                       
            self.feature_extraction = encoder
            if self.patch2feat:
                self.transition_layer = patch2feature(
                        embed_dim=768, patch_size=14, output_dim=pred_dim,
                        out_channels=(48, 96, 192, 384),
                    )
            else:
                self.transition_layer = DinoUpsampler(
                    embed_dim=768, patch_size=14,
                    output_dim=pred_dim, num_layers=4,
                    upsample_factor=4,
                )
            self.feat_channel = pred_dim
        
        else:
            self.feature_extraction = efficientnet_feature(self.stereo) 
            self.feat_channel = self.feature_extraction.feat_channel
        if regression:
                                     
                                          
            self.ele_head = EleReg2D(self.feat_channel, num_grids, normalize)

        else:
            if self.stereo:
                                       
                self.ele_head = EleCla3D(self.feat_channel, num_grids, self.num_classes)
            else:
                                     
                self.ele_head = EleCla2D(self.feat_channel, num_grids, self.num_classes)

        if 'DepthAnything3' in backbone:
            self.transition_layer.apply(self._init_weights)
            self.ele_head.apply(self._init_weights)
        else:
                                                  
            self.feature_extraction.apply(self._init_weights)
            self.ele_head.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
                                                   
                      
    def forward(self, imgs_left, proj_index_left, *args):
                                                                           
        if 'DepthAnything3' in self.backbone:
            encoder_ctx = nullcontext() if self.train_encoder else torch.no_grad()
            with encoder_ctx:
                                                                          
                                            
                B, C, W, H = imgs_left.shape
                imgs_left = imgs_left.unsqueeze(1)                  
                                                        
                print("me", imgs_left.shape)
                features, _ = self.feature_extraction(imgs_left)                                                                
                B, S, N, C = features[0][0].shape
                print("Extracted features before projection shape:", features[0][0].shape)
                features = [feat[0].reshape(B*S, N, C) for feat in features]
                                                                                                                                 
                                      
            if self.patch2feat:
                features_left = self.transition_layer(features, W, H, int(W/4), int(H/4))                    
            else:
                features_left = self.transition_layer(features, W, H, W/4, H/4)                  
                                                                   
        
        else:
                                                                        
            features_left = self.feature_extraction(imgs_left)
                                                                    
                                        
        B, C, H, W = features_left.shape
        self._last_features = features_left.detach()
        features_left = features_left.reshape(B, C, -1)
        linear_indices = proj_index_left[:, 1, :] * W + proj_index_left[:, 0, :]

        voxel_feat_left = features_left.gather(dim=2, index=linear_indices.unsqueeze(1).expand(-1, C, -1))
                                                                       

        voxel_feat_left = voxel_feat_left.reshape(B, C, self.num_grids_z, self.num_grids_x, self.num_grids_y)
                                                                          
                                                                                                      
        if self.stereo:
            imgs_right, proj_index_right = args[0], args[1]
            features_right = self.feature_extraction(imgs_right)
            features_right = features_right.reshape(B, C, -1)
            linear_indices = proj_index_right[:, 1, :] * W + proj_index_right[:, 0, :]
            voxel_feat_right = features_right.gather(dim=2, index=linear_indices.unsqueeze(1).expand(-1, C, -1))
            voxel_feat_right = voxel_feat_right.reshape(B, C, self.num_grids_z, self.num_grids_x, self.num_grids_y)

            voxel_feature = voxel_feat_left * voxel_feat_right
            voxel_feature = voxel_feature.permute(0, 1, 4, 2, 3)                   
        else:
            voxel_feature = voxel_feat_left                     

        ele_pred = self.ele_head(voxel_feature)                                            

        if (not self.training) & (not self.regression):
            ele_pred = F.softmax(ele_pred, dim=1)
            ele_pred = torch.sum(ele_pred * self.ele_values, dim=1)

                                                         
        return ele_pred


class DinoV2SpatialDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int = 14,
        out_channels: Optional[int] = None,
        intermediate_layer_idx=(0, 1, 2, 3),
    ):
        super().__init__()

        self.patch_size = patch_size
        self.intermediate_layer_idx = intermediate_layer_idx

                                                     
        self.out_channels = out_channels or embed_dim

                              
        self.projects = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 8, self.out_channels, kernel_size=1))
            
            for _ in intermediate_layer_idx
        ])

                             
        self.fuse = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
        )

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )

        self.fpn3 = nn.Identity()

        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.norm = nn.LayerNorm(embed_dim)


    def forwardää(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        H_out: int,
        W_out: int,
        patch_start_idx: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            feats: list of 4 tensors, each [B, N, C]
            H, W: target spatial resolution

        Returns:
            Tensor: (B, C, H, W)
        """
        assert len(feats) == len(self.intermediate_layer_idx)

                                                            
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size

        resized_feats = []

        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]                                
                                                                           
                                                                                    
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)                 
                                                                             
                              
            x = self.projects[stage_idx](x)                            

                                                                            
            x = F.interpolate(
                x,
                size=(H_out, W_out),
                mode="bilinear",
                align_corners=False,
            )                          

            resized_feats.append(x)

                                   
        fused = torch.stack(resized_feats, dim=0)                             
        fused = fused.sum(dim=0)
        fused = self.fuse(fused) 
                                                                      
                                                                              
        return fused

    def forward(
            self,
            feats: List[torch.Tensor],
            H: int,
            W: int,
            H_out: int,
            W_out: int,
            patch_start_idx: int = 0,
    ) -> torch.Tensor:
        features =[]
        feats = [feats[i][:, patch_start_idx:] for i in range(len(feats))]
        ph, pw = H // self.patch_size, W // self.patch_size 
        feats = [
            feat.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[2], ph, pw) 
            for feat in feats
        ]                  
                                                             
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        if len(feats) > 1:
            for i in range(len(ops)):
                features.append(feats[-1])
            for i in range(len(features)):
                features[i] = ops[i](features[i])
                features[i] = self.projects[i](features[i])
                                                                              
                features[i] = F.interpolate(
                    features[i],
                    size=(H_out, W_out),
                    mode="bilinear",
                    align_corners=False,
                )                           
            
            features_fused = torch.stack(features, dim=0).sum(dim=0)
                                                                            

        return features_fused
            
            
