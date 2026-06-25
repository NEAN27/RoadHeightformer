import math
import torch, sys
import torch.nn as nn
from typing import List, Sequence, Tuple, Union, Optional
from .DPT_utils import Permute
from utils.experiment import save_feature_map
from models.submodule import *
from sklearn.decomposition import PCA
import cv2

sys.path.append('/home/f9ql00v/depth-anything3/Depth-Anything-3-main/src')
from depth_anything_3.model.utils.head_utils import (
    create_uv_grid,
    position_grid_to_embed,
)

def stat(x, name):
    print(
        name,
        "mean", x.mean().item(),
        "std", x.std().item(),
        "max", x.abs().max().item()
    )

class patch2feature(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int = 256, 
        patch_size: int = 16,
        out_channels:  Sequence[int] = (256, 512, 1024, 1024),
        intermediate_layer_idx=(0, 1, 2, 3),
        pos_embed: bool = False, 
    ):
        super().__init__()
        self.patch_size = patch_size
        self.intermediate_layer_idx = intermediate_layer_idx
        self.pos_embed = pos_embed
                                                     
        self.out_channels = out_channels or embed_dim

        self.projects = nn.ModuleList(
            nn.Conv2d(embed_dim, oc, kernel_size=1, stride=1, padding=0, bias=True) for oc in out_channels
                                                                                                       
        )

                                                                                                         
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0                       
                ),
                nn.ConvTranspose2d(
                    out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0                      
                ),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),                            
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

                                                                                                
        self.scratch = _make_scratch(list(out_channels), output_dim, expand=False)
        self.scratch.output_conv1 = nn.Conv2d(
            output_dim, output_dim, kernel_size=3, stride=1, padding=1
        )
                                
        self.out_norm = nn.Sequential(  
            Permute((0, 2, 3, 1)), nn.LayerNorm(output_dim), Permute((0, 3, 1, 2)),
        )

                           
        self.scratch.refinenet1 = _make_fusion_block(output_dim, inplace=False)
        self.scratch.refinenet2 = _make_fusion_block(output_dim, inplace=False)
        self.scratch.refinenet3 = _make_fusion_block(output_dim, inplace=False)
        self.scratch.refinenet4 = _make_fusion_block(
            output_dim, has_residual=False, inplace=False
        )

        self.rn_norm = nn.ModuleList([
            nn.GroupNorm(32, output_dim),
            nn.GroupNorm(32, output_dim),
            nn.GroupNorm(32, output_dim),
            nn.GroupNorm(32, output_dim)
        ])

        self.fuse_norm = nn.ModuleList([
                nn.GroupNorm(32, output_dim),                    
                nn.GroupNorm(32, output_dim),                    
                nn.GroupNorm(32, output_dim),                    
                nn.GroupNorm(32, output_dim),                    
        ])

    def _add_pos_embed(
        self,
        x: torch.Tensor,
        W: int,
        H: int,
        ratio: float = 0.1,
    ) -> torch.Tensor:
        """
        UV positional embedding (same as DualDPT).
        """
        pw, ph = x.shape[-1], x.shape[-2]

        pe = create_uv_grid(
            pw,
            ph,
            aspect_ratio=W / H,
            dtype=x.dtype,
            device=x.device,
        )

        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)

        print(pe.shape, x.shape)
        return x + pe
    
    def _fuse(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        4-layer top-down fusion, returns finest scale features (after fusion, before neck1).
        """
        l1, l2, l3, l4 = feats

        l1_rn = self.scratch.layer1_rn(l1)
                                              
                                                 
        l2_rn = self.scratch.layer2_rn(l2)
                                                                                                               
                                                 
        l3_rn = self.scratch.layer3_rn(l3)
                                                                                                               
        l4_rn = self.scratch.layer4_rn(l4)
                           
                                              
        l1_rn = self.rn_norm[0](self.scratch.layer1_rn(l1))
        l2_rn = self.rn_norm[1](self.scratch.layer2_rn(l2))
        l3_rn = self.rn_norm[2](self.scratch.layer3_rn(l3))
        l4_rn = self.rn_norm[3](self.scratch.layer4_rn(l4))
                          
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.fuse_norm[0](out)
                                                
                           
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.fuse_norm[1](out)
                           

        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.fuse_norm[2](out)
                                                
                                           
        out = self.scratch.refinenet1(out, l1_rn)
        out = self.fuse_norm[3](out)
                           
        return out
    
    def forward(self, feats: List[torch.Tensor],
                H: int,
                W: int,
                h_out,
                w_out,
                patch_start_idx: int = 0) -> torch.Tensor:
                                                                                  
                                                                             
        if len(feats) == 1:
            feats = [feats[0]] * 4
                                       
                                           
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
                                                                       
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]                     

            x = self.norm(x)
                                            
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)                          
                                                 
                                     
            x = self.projects[stage_idx](x)                                               
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
                                                                                                     

            x = self.resize_layers[stage_idx](x)               
                                               

            resized_feats.append(x)

                                              
        fused = self._fuse(resized_feats)
        if self.pos_embed:
            fused = self._add_pos_embed(fused, W, H)
                                              
                                                                                                                    
        fused = self.scratch.output_conv1(fused)
                                                                                               
                                    
        fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)
                                                                 

        return fused

def _make_scratch(
    in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False
) -> nn.Module:
    scratch = nn.Module()
                                 
    c1 = out_shape
    c2 = out_shape * (2 if expand else 1)
    c3 = out_shape * (4 if expand else 1)
    c4 = out_shape * (8 if expand else 1)

    scratch.layer1_rn = nn.Conv2d(in_shape[0], c1, 3, 1, 1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], c2, 3, 1, 1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], c3, 3, 1, 1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], c4, 3, 1, 1, bias=False, groups=groups)
    return scratch

def _make_fusion_block(
    features: int,
    size: Tuple[int, int] = None,
    has_residual: bool = True,
    groups: int = 1,
    inplace: bool = False,
) -> nn.Module:
    return FeatureFusionBlock(
        features=features,
        activation=nn.ReLU(inplace=inplace),
        deconv=False,
        bn= False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )

class ResidualConvUnit(nn.Module):
    """Lightweight residual convolution block for fusion"""

    def __init__(self, features: int, activation: nn.Module, bn: bool, groups: int = 1) -> None:
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        if bn:
                                           
            self.norm1 = nn.BatchNorm2d(features)
            self.norm2 = nn.BatchNorm2d(features)
        else:
            self.norm1 = None
            self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:                          
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
                                
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)

                                                                               
def custom_interpolate(
    x: torch.Tensor,
    size: Union[Tuple[int, int], None] = None,
    scale_factor: Union[float, None] = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Safe interpolation implementation to avoid INT_MAX overflow in torch.nn.functional.interpolate.
    """
    if size is None:
        assert scale_factor is not None, "Either size or scale_factor must be provided."
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736
    total = size[0] * size[1] * x.shape[0] * x.shape[1]

    if total > INT_MAX:
        chunks = torch.chunk(x, chunks=(total // INT_MAX) + 1, dim=0)
        outs = [
            nn.functional.interpolate(c, size=size, mode=mode, align_corners=align_corners)
            for c in chunks
        ]
        return torch.cat(outs, dim=0).contiguous()

    return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)

def make_pca(out, name, feat_dim):
    C, H, W = out[0].shape
    out_pca = out[0].cpu().detach().permute(1, 2, 0).reshape(-1, C)
    
    print("out 2nd refinenet: ", out_pca.shape)
    pca = PCA(n_components=3)
    pca.fit(out_pca)
    pca_features = pca.transform(out_pca)
    pca_features = pca_features.reshape(H, W, 3)
    
    min_vals = pca_features.min(axis=2, keepdims=True)
    max_vals = pca_features.max(axis=2, keepdims=True)

    pca_features = (pca_features - min_vals) / (max_vals - min_vals + 1e-8)
    pca_features = (pca_features * 255).astype(np.uint8)
    cv2.imwrite(name, pca_features)
    
class FeatureFusionBlock(nn.Module):
    """Top-down fusion block: (optional) residual merge + upsampling + 1x1 contraction"""

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size: Tuple[int, int] = None,
        has_residual: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.size = size
        self.has_residual = has_residual
        self.resConfUnit1 = (
            ResidualConvUnit(features, activation, bn, groups=groups) if has_residual else None
        )
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=groups)

        out_features = (features // 2) if expand else features
        self.out_conv = nn.Conv2d(features, out_features, 1, 1, 0, bias=True, groups=groups)
        
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs: torch.Tensor, size: Tuple[int, int] = None) -> torch.Tensor:                          
        """
        xs:
          - xs[0]: Top branch input
          - xs[1]: Lateral input (can do residual addition with top branch)
        """
        y = xs[0]
        if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
            y = self.skip_add.add(y, self.resConfUnit1(xs[1]))

        y = self.resConfUnit2(y)

                    
        if (size is None) and (self.size is None):
            up_kwargs = {"scale_factor": 2}
        elif size is None:
            up_kwargs = {"size": self.size}
        else:
            up_kwargs = {"size": size}

        y = custom_interpolate(y, **up_kwargs, mode="bilinear", align_corners=self.align_corners)
        y = self.out_conv(y)
        return y


class easy_transition_layer(nn.Module):
    def __init__(self,
        embed_dim: int,
        patch_size: int = 14,
        out_channels: Optional[int] = None,
        intermediate_layer_idx=(0, 1, 2, 3),
    ):
        super().__init__()

        self.patch_size = patch_size
        self.intermediate_layer_idx = intermediate_layer_idx

                                                     
        self.out_channels = out_channels or embed_dim
        self.norm = nn.Identity()                         
                              
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
    

    def forward(self, feats: List[torch.Tensor], H: int, W: int, H_out: int, W_out: int) -> torch.Tensor:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        x = feats[-1]                     
        x = self.norm(x)
                                                                                       
        x = x.permute(0, 2, 1).reshape(B, C, ph, pw)                    

        x = self.projects[-1](x)
                                                                                                 

        x = custom_interpolate(x, size=(H_out, W_out), mode="bilinear", align_corners=True)
                                                                                                    

        return x
    

def visualize_value(fused, name_of_file):
    x = fused
    idx = torch.argmax(x)

                            
    coords = torch.unravel_index(idx, x.shape)

                                                      
    if x.shape[1] < 82:
        None
                                                                 
    else:
        None
                                                     

class DinoUpsampler(nn.Module):
    """
    Token -> dense feature map upsampler for ViT backbones (e.g. DINOv2).

    Designed as a lighter, quality-preserving drop-in for `patch2feature`:
      - keeps channel width near embed_dim until the very end,
      - bilinear-resize-then-conv upsampling (no transposed-conv checkerboards),
      - per-token LayerNorm matching DINOv2 normalisation,
      - learned softmax-weighted fusion across the L selected ViT layers,
      - residual bilinear skip from the deepest token map ('FeatUp-lite').

    Forward signature mirrors `patch2feature.forward` so it can be swapped in
    without changing call sites:
        out = up(feats, H, W, h_out, w_out)

    Args (forward):
        feats : list of L tensors, each [B, N, embed_dim] (CLS already stripped).
        H, W  : spatial size of the input image (used to derive ph, pw).
        h_out, w_out : target dense-map size.
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: int = 256,
        patch_size: int = 14,
        num_layers: int = 4,
        upsample_factor: int = 4,
        hidden_dim: Optional[int] = None,
        residual_skip: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.upsample_factor = upsample_factor
        self.residual_skip = residual_skip
        hidden_dim = hidden_dim or embed_dim

        self.token_norms = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(num_layers)]
        )
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        self.mix = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU(),
        )

        n_stages = int(round(math.log2(float(upsample_factor))))
        assert 2 ** n_stages == upsample_factor, "upsample_factor must be a power of 2"
        self.upsample_blocks = nn.ModuleList()
        for _ in range(n_stages):
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(32, hidden_dim),
                    nn.GELU(),
                )
            )

        self.proj = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def _tokens_to_map(self, tokens: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
        B, N, C = tokens.shape
        assert N == ph * pw, f"token count {N} != ph*pw {ph*pw}"
        return tokens.transpose(1, 2).reshape(B, C, ph, pw)

    def forward(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        h_out: int,
        w_out: int,
        patch_start_idx: int = 0,
    ) -> torch.Tensor:
        assert len(feats) == self.num_layers
        feats = [f[:, patch_start_idx:] for f in feats]
        ph, pw = H // self.patch_size, W // self.patch_size

        weights = torch.softmax(self.layer_weights, dim=0)
        fused_tokens = sum(
            w * self.token_norms[i](feats[i]) for i, w in enumerate(weights)
        )

        x = self._tokens_to_map(fused_tokens, ph, pw)
        x = self.mix(x)

        for block in self.upsample_blocks:
            x = block(x)

        if self.residual_skip:
            deep = self._tokens_to_map(self.token_norms[-1](feats[-1]), ph, pw)
            deep = nn.functional.interpolate(
                deep, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            x = x + self.mix(deep)

        if x.shape[-2:] != (h_out, w_out):
            x = nn.functional.interpolate(
                x, size=(h_out, w_out), mode="bilinear", align_corners=False
            )

        return self.proj(x)