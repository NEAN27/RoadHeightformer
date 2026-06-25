
import torch.nn as nn
import torch.nn.functional as F
from models.submodule import *
from .efficientnet import efficientnet_cla
from utils.experiment import save_feature_map

class EleCla2D(nn.Module):
    def __init__(self, feat_channel, num_grids, num_classes):
        super(EleCla2D, self).__init__()
        self.num_grids_x, self.num_grids_y, self.num_grids_z = num_grids
        self.feat_channel = feat_channel
        self.channel_reshaped = self.feat_channel * self.num_grids_y
        self.num_classes = num_classes
        print("EleCla2D: channel_reshaped:", self.feat_channel, "*", self.num_grids_y, "=", self.channel_reshaped)

        self.inplanes = int(self.channel_reshaped/8)
        self.first_conv = nn.Sequential(
                        convbn(self.channel_reshaped, self.inplanes, 5, 1, 2, 1),
                        nn.ReLU(inplace=True))

        self.effnet_reg = efficientnet_cla(self.inplanes, self.num_classes)

        self.final_conv = nn.Sequential(convbn(self.num_classes, self.num_classes, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(self.num_classes, self.num_classes, kernel_size=1, stride=1,
                                                  padding=0, bias=False))

    def forward(self, feat_voxel):
                                     
        B = feat_voxel.shape[0]
        print("EleCla2D: feat_voxel shape:", feat_voxel.shape)
                                                                           
        feat_bev = feat_voxel.permute(0, 4, 1, 2, 3)
        feat_bev = feat_bev.reshape(B, self.channel_reshaped, self.num_grids_z, self.num_grids_x)               
        feat_bev = self.first_conv(feat_bev)
        feat_bev = self.effnet_reg(feat_bev)
        ele_cla_prob = self.final_conv(feat_bev)                          

        return ele_cla_prob

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1), nn.ReLU(inplace=True))
        self.attention_block = attention_block(channels_3d=in_channels * 2, num_heads=16, block=(2,2,2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)       
        conv2 = self.conv2(conv1)       
        conv3 = self.attention_block(conv2)

        conv6 = F.relu(self.conv6(conv3) + self.redir1(x), inplace=True)
        return conv6

class EleCla3D(nn.Module):
    def __init__(self, feat_channel, num_grids, num_classes):
        super(EleCla3D, self).__init__()
        self.feat_channel = feat_channel
        self.num_grids_x, self.num_grids_y, self.num_grids_z = num_grids
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            convbn_3d(self.feat_channel, self.feat_channel, kernel_size=(3, 5, 5), stride=1, pad=(1, 2, 2)),      
            nn.ReLU(inplace=True),
            hourglass(self.feat_channel),
            convbn_3d(self.feat_channel, self.feat_channel, kernel_size=(3, 5, 5), stride=1, pad=(1, 2, 2)),
            nn.ReLU(inplace=True),
            convbn_3d(self.feat_channel, int(self.feat_channel / 2), 3, 1, 1),
            nn.ReLU(inplace=True),
            hourglass(int(self.feat_channel / 2)),
            convbn_3d(int(self.feat_channel / 2), int(self.feat_channel / 2), 3, 1, 1),
            nn.ReLU(inplace=True),
            hourglass(int(self.feat_channel / 2)),
            convbn_3d(int(self.feat_channel / 2), int(self.feat_channel / 2), 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(self.feat_channel / 2), 1, kernel_size=3, padding=1, stride=1, bias=False)
            )

    def forward(self, feat_voxel):
                                     
        prob_volume = self.layer1(feat_voxel)                                         
        prob_volume = F.interpolate(prob_volume, [self.num_classes, prob_volume.size()[3], prob_volume.size()[4]], mode='trilinear',
                              align_corners=True)
        prob_volume = prob_volume.squeeze(1)

        return prob_volume

class EleReg2D(nn.Module):
    def __init__(self, feat_channel, num_grids, normalize):
        super(EleReg2D, self).__init__()
        self.num_grids_x, self.num_grids_y, self.num_grids_z = num_grids
        self.channel_reshaped = feat_channel * self.num_grids_y

        self.inplanes = int(self.channel_reshaped / 8)

        self.reg_head = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes // 2, 3, padding=1, bias=True),
            nn.BatchNorm2d(self.inplanes // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.inplanes // 2, self.inplanes // 4, 3, padding=1, bias=True),
            nn.BatchNorm2d(self.inplanes // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.inplanes // 4, 1, kernel_size=1, bias=True)
        )
        self.first_conv = nn.Sequential(
            convbn(self.channel_reshaped, self.inplanes, 3, 1, 1, 1),                                                                        
            nn.ReLU(inplace=True)
        )
                       
                                                                              
    def forward(self, feat_voxel):
                                     
        B = feat_voxel.shape[0]
                                                               
        feat_bev = feat_voxel.permute(0, 4, 1, 2, 3)
        feat_bev = feat_bev.reshape(
            B, self.channel_reshaped, self.num_grids_z, self.num_grids_x
        )
                                                                                                    
                                                                    
        feat_bev = self.first_conv(feat_bev)
                                                                                           
                                                                              
        ele = self.reg_head(feat_bev)
                                                                     
        if hasattr(self, 'normoutput'):
            print("helllaaa")
            ele = self.normoutput(ele)           
        """
        feat_bev = self.effnet_reg(feat_bev)
        ele = self.final_conv(feat_bev) """
                               
        return ele.squeeze(1)             