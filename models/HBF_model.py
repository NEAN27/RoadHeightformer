from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.ele_head import *
import math
from .efficientnet import efficientnet_feature
from mmcv.cnn.bricks.transformer import build_attention
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time

class Elevation(nn.Module):
    def __init__(self, stereo,  num_grids, ele_range, cla_res):
        super(Elevation, self).__init__()
        #*****not useful yet for HBF*****
        self.stereo = stereo
        self.num_grids_x, self.num_grids_y, self.num_grids_z = num_grids
        self.ele_range = ele_range   # in meter

        self.cla_res = cla_res
        self.num_classes = int(2 * self.ele_range*100 / self.cla_res)
        ele_values = -torch.arange(self.num_classes, dtype=torch.float32, device='cuda')*self.cla_res + self.ele_range*100 - self.cla_res/2
        self.ele_values = ele_values.reshape(1, self.num_classes, 1, 1)
        #*******************************

        # Replace efficientnet_feature with DINOv2 backbone
        self.feature_extraction = self._initialize_dinov2_backbone() # efficientnet_feature(self.stereo)
        #self.feature_extraction = DepthAnythingBackbone(self.stereo)

        if self.stereo:
            #  regressor for stereo
            self.ele_head = EleCla3D(self.feature_extraction.feat_channel, num_grids, self.num_classes)
        else:
            #  regressor for mono
            self.ele_head = EleCla2D(self.feature_extraction.feat_channel, num_grids, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
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

    def _initialize_dinov2_backbone(self):
        # Load pretrained DINOv2 backbone
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        dinov2.eval()  # Set to evaluation mode
        dinov2.feat_channel = 768  # DINOv2 base feature map channel count
        return dinov2
    
    def forward(self, imgs_left, proj_index_left, *args):
        # proj_index: [num_samples, 2, num_grids_z*num_grids_x*num_grids_y]
        #Extract three features
        with torch.no_grad():
            features_left = self.feature_extraction(imgs_left)
            print("Extracted features shape:", features_left.shape)
        
        #Use attention to compute refine the ele prediction
        

        return ele_pred
