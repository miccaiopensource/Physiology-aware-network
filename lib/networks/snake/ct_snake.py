import torch.nn as nn
import torch.nn.functional as F
from .dla import DLASeg
from .evolve import Evolution
from lib.utils import net_utils, data_utils
from lib.utils.snake import snake_decode
from lib.networks.snake.snake import Thickness_Classifier
import torch
# from .hybrid.models.hybrid_res_unet import ResUNet18_5 as ResUNet
from .hybrid.models.hybrid_res_unet import ResUNet18 as ResUNet
# from .hybrid.models.hybrid_res_unet import ResUNet18_64out as ResUNet
from .hybrid.models.hybrid_res_unet import ResUNet18_tiny as ResUNet_t
from .hybrid.models.hybrid_res_unet import ResUNet18_Cls as ResUNet_cls
from .hybrid.models.hybrid_res_unet import ResUNet18_3Cls as ResUNet_3cls
from .hybrid.models.hybrid_res_unet import SkipResUNet18_3Cls as SkipResUNet_3cls
# from .hybrid.models.res_unet import ResUNet_cls3d as ResUNet_3cls
from .hybrid.models.hybrid_res_unet import ResUNet18_block as ResUNet_block
from .hybrid.models.hybrid_res_unet import ResUNet_block_32_3 as ResUNet_block_3out
from .hybrid.models.hybrid_res_unet import ResUNet_block_32_2 as ResUNet_block_2out
# from .hybrid.models.hybrid_res_unet import ResUNet18_Double_Skip as ResUNet_2
from .hybrid.models.hybrid_res_unet import ResUNet18_Double_Skip as ResUNet_2
from .hybrid.models.hybrid_res_unet import Time_Series as Time_Series
from .hybrid.models.hybrid_res_unet import ResTime_Series as ResTime_Series
# from .hybrid.models.hybrid_res_unet import ResUNet18_Double as ResUNet_2
import json
import sys
import numpy as np

from lib.utils import img_utils, data_utils
from lib.utils.snake import snake_config
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import cycle
import os
from termcolor import colored
import SimpleITK as sitk

mean = snake_config.mean
std = snake_config.std
import nibabel as nib
import pandas as pd
import numpy as np
import scipy
from scipy import ndimage
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import skimage
from skimage import feature
from scipy import spatial

class Network(nn.Module):

    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
        super(Network, self).__init__()
        # self.Timeseries = Time_Series()
        # self.Timeseries = ResTime_Series()
        # self.hybrid = ResUNet()
        self.hybrid_1 = ResUNet_2()
        self.hybrid_2 = ResUNet_2()
        # # self.hybrid_1 = ResUNet()
        # # self.hybrid_2 = ResUNet()
        # # self.hybrid_cls = ResUNet_cls()
        # self.hybrid_cls = ResUNet_3cls()
        self.hybrid_cls = SkipResUNet_3cls()
        self.hybrid_block = ResUNet_block_3out()
        self.hybrid_block_seg = ResUNet_block_2out()
        # self.hybrid_block_seg = ResUNet_block_seg_2out()


        # self.dla = DLASeg('dla{}'.format(num_layers), heads,
        #                   pretrained=False,
        #                   down_ratio=down_ratio,
        #                   final_kernel=1,
        #                   last_level=5,
        #                   head_conv=head_conv)
        self.gcn = Evolution()
        # self.thickness = Thickness_Classifier(state_dim=128)

    def decode_detection(self, output, h, w):
        ct_hm = output['ct_hm']
        wh = output['wh']
        ct, detection = snake_decode.decode_ct_hm(torch.sigmoid(ct_hm), wh)
        detection[..., :4] = data_utils.clip_to_image(detection[..., :4], h, w)
        output.update({'ct': ct, 'detection': detection})
        return ct, detection

    def forward(self, x, batch=None):
        thickness_set = batch['thickness_set']
        thickness_set=thickness_set.type(torch.cuda.FloatTensor)

        output= self.thickness(thickness_set)
        output_f={}
        output_f["prob"]=output

        return output_f

    # Bayesian Uncertainty ellipse
    def forward(self, x, batch=None):
        # output, cnn_feature = self.dla(x)
        # print("input_shape:",x.shape)
        # input_shape: torch.Size([18, 15, 192, 192])
        # input_shape: torch.Size([18, 15, 192, 192])
        # input_shape: torch.Size([18, 15, 192, 192])
        # input_shape: torch.Size([18, 15, 192, 192])
        # sys.exit()
        output_f = {}
        x = torch.unsqueeze(x, 1)
        # output, cnn_feature = self.hybrid(x)
        prob_health_t,_,seg_from_cls,_ = self.hybrid_cls(x)
        # prob_health_t,plq_seg_from_cls = self.hybrid_cls(x)
        prob_health = prob_health_t.detach()
        # prob_health = self.hybrid_cls(x)[:,7,:]
        # print("prob_shape:",prob_health.shape)
        # sys.exit()
        # print("before hybrid1")

        output_h, cnn_feature_h,output_h_seg,cnn_feature_h_seg = self.hybrid_1(x)

        # print("cnn_feature_h:",cnn_feature_h.shape)
        # output_h, cnn_feature_h,output_seg_h_0,output_seg_h_1,cnn_feature_h_seg = self.hybrid_1(x)

        output_unh, cnn_feature_unh,output_seg_unh,cnn_feature_unh_seg = self.hybrid_2(x)

        # print("F.softmax(prob_health)[:, 0]:",F.softmax(prob_health)[:, 0].shape)
        # print("F.softmax(prob_health)[:, 0].reshape(-1, 1, 1, 1):", F.softmax(prob_health)[:, 0].reshape(-1, 1, 1, 1).shape)
        # sys.exit()
        output = torch.mul(F.softmax(prob_health)[:, 0].reshape(-1, 1, 1, 1).expand(-1, 32, 96, 96),
                           cnn_feature_h) \
                 + torch.mul(F.softmax(prob_health)[:, 1].reshape(-1, 1, 1, 1).expand(-1, 32, 96, 96),
                             cnn_feature_h) \
                 + torch.mul(F.softmax(prob_health)[:, 2].reshape(-1, 1, 1, 1).expand(-1, 32, 96, 96),
                             cnn_feature_unh)
        output_seg = torch.mul(F.softmax(prob_health)[:, 0].reshape(-1, 1, 1, 1).expand(-1, 32, 96, 96),
                            cnn_feature_h_seg) \
                    + torch.mul(F.softmax(prob_health)[:, 1].reshape(-1, 1, 1, 1).expand(-1, 32, 96, 96),
                                cnn_feature_h_seg) \
                    + torch.mul(F.softmax(prob_health)[:, 2].reshape(-1, 1, 1, 1).expand(-1, 32, 96, 96),
                                cnn_feature_unh_seg)

        # output_seg =  torch.mul(F.softmax(prob_health)[:, 2].reshape(-1, 1, 1, 1).expand(-1, 32, 96, 96),
        #                         cnn_feature_unh_seg)


        output,cnn_feature = self.hybrid_block(output)
        output_seg, cnn_feature_seg = self.hybrid_block_seg(output_seg)




        output_f["prob_health"] =F.softmax(prob_health_t, 1)
        # output_f["seg_from_cls"] =F.softmax( seg_from_cls,1)
        output_f["prob_map"] = output
        output_f["prob_map_seg"] =F.softmax(output_seg,1)



        output_f = self.gcn(output_f, cnn_feature,cnn_feature_seg, batch)
        return output_f






def get_network(num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
    network = Network(num_layers, heads, head_conv, down_ratio, det_dir)
    return network
