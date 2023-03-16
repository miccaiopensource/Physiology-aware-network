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
    # def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
    #     super(Network, self).__init__()
    #     # self.Timeseries = Time_Series()
    #     self.Timeseries = ResTime_Series()
    #     # self.hybrid = ResUNet()
    #     # self.hybrid_2 = ResUNet_2()
    #     # self.hybrid_1 = ResUNet()
    #     # self.hybrid_2 = ResUNet()
    #     # self.hybrid_cls = ResUNet_cls()
    #     # self.hybrid_cls = ResUNet_3cls()
    #     # self.hybrid_block = ResUNet_block()
    #
    #
    #     # self.dla = DLASeg('dla{}'.format(num_layers), heads,
    #     #                   pretrained=False,
    #     #                   down_ratio=down_ratio,
    #     #                   final_kernel=1,
    #     #                   last_level=5,
    #     #                   head_conv=head_conv)
    #     self.gcn = Evolution()
    #     # self.thickness = Thickness_Classifier(state_dim=128)
    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
        super(Network, self).__init__()
        # self.Timeseries = Time_Series()
        # self.Timeseries = ResTime_Series()
        self.hybrid = ResUNet()
        # self.hybrid_1 = ResUNet_2()
        # self.hybrid_2 = ResUNet_2()
        # # self.hybrid_1 = ResUNet()
        # # self.hybrid_2 = ResUNet()
        # # self.hybrid_cls = ResUNet_cls()
        # self.hybrid_cls = ResUNet_3cls()
        # # self.hybrid_cls = SkipResUNet_3cls()
        # self.hybrid_block = ResUNet_block_3out()
        # self.hybrid_block_seg = ResUNet_block_2out()
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
        #
        # print('thickness_set:',thickness_set.shape)

        # output, cnn_feature = self.hybrid(x)
        output= self.thickness(thickness_set)
        output_f={}
        output_f["prob"]=output

        return output_f

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
        output, cnn_feature = self.hybrid(x)
        # prob_health = self.hybrid_cls(x)
        # # prob_health = self.hybrid_cls(x)[:,7,:]
        # # print("prob_shape:",prob_health.shape)
        # # sys.exit()
        # output_health = self.hybrid_1(x)
        # output_unhealth = self.hybrid_2(x)
        # print("F.softmax(prob_health)[:, 0]:",F.softmax(prob_health)[:, 0].shape)
        # print("F.softmax(prob_health)[:, 0].reshape(-1, 1, 1, 1):", F.softmax(prob_health)[:, 0].reshape(-1, 1, 1, 1).shape)
        # sys.exit()
        # output = torch.mul(F.softmax(prob_health)[:, 0].reshape(-1, 1, 1, 1).expand(-1, 64, 96, 96),
        #                    output_health) \
        #          + torch.mul(F.softmax(prob_health)[:, 1].reshape(-1, 1, 1, 1).expand(-1, 64, 96, 96),
        #                      output_health) \
        #          + torch.mul(F.softmax(prob_health)[:, 2].reshape(-1, 1, 1, 1).expand(-1, 64, 96, 96),
        #                      output_unhealth)
        # output = torch.mul(F.softmax(prob_health)[:, 0].reshape(-1, 1, 1, 1).expand(-1, 64, 96, 96),
        #                                  output_health) \
        #                        + torch.mul(F.softmax(prob_health)[:, 1].reshape(-1, 1, 1, 1).expand(-1, 64, 96, 96),
        #                                    output_health) \
        #                        + torch.mul(F.softmax(prob_health)[:, 2].reshape(-1, 1, 1, 1).expand(-1, 64, 96, 96),
        #                                    output_unhealth)\
        #          + torch.mul(F.softmax(prob_health)[:, 3].reshape(-1, 1, 1, 1).expand(-1, 64, 96, 96),
        #                                    output_unhealth)
        # output_f["prob_health"] = F.softmax(prob_health)
        # output_f["output_health"] = output_health
        # output_f["output_unhealth"] = output_unhealth
        # output,cnn_feature = self.hybrid_block(output)
        #
        # output_f["prob_health"] =F.softmax(prob_health)
        output_f["prob_map"] = output

        # print("prob_health[:,0]:",prob_health[:,0].reshape(-1,1,1,1).shape)
        # print("output_health:", output_health.shape)
        # print("cnn_feature_h:", cnn_feature_h.shape)
        # print("prob_health[:,0].expand(-1,64, 96, 96):",F.softmax(prob_health)[:,0].reshape(-1,1,1,1).expand(-1,2, 2, 2))
        # print("prob_health[:,0].expand(-1,64, 96, 96):", F.softmax(prob_health)[:, 1].reshape(-1, 1, 1, 1).expand(-1, 3, 2, 2))
        # sys.exit()

        # output_f["prob_map"] = torch.mul(F.softmax(prob_health)[:,0].reshape(-1,1,1,1).expand(-1,3, 96, 96),output_health) \
        #                        +torch.mul(F.softmax(prob_health)[:,1].reshape(-1,1,1,1).expand(-1,3, 96, 96),output_unhealth)\
        #                        +torch.mul(F.softmax(prob_health)[:,2].reshape(-1,1,1,1).expand(-1,3, 96, 96),output_unhealth)
        # cnn_feature = torch.mul(F.softmax(prob_health)[:,0].reshape(-1,1,1,1).expand(-1,64, 96, 96),cnn_feature_h)\
        #               +torch.mul(F.softmax(prob_health)[:,1].reshape(-1,1,1,1).expand(-1,64, 96, 96),cnn_feature_unh)\
        #               +torch.mul(F.softmax(prob_health)[:,2].reshape(-1,1,1,1).expand(-1,64, 96, 96),cnn_feature_unh)


        # cnn_feature = prob_health[:0]*cnn_feature_h +prob_health[:1]*cnn_feature_unh
        # print("output_size:",output.shape)
        # print("cnn_feature_size:", cnn_feature.shape)
        # print("mask_shape:",batch["mask"].shape)
        # mask_shape: torch.Size([3, 192, 192])
        #
        # # sys.exit()
        # inp = img_utils.unnormalize_img(batch['inp'][0][15 // 2, :, :], mean[:, :, 15 // 2], std[:, :, 15 // 2])
        # output_test= F.softmax(output_f["prob_map"],dim=1)[-1].cpu().numpy()
        # inner=output_test[1]*255
        # inner_f=np.zeros([96,96])
        # inner_f[inner>=0.5]=1
        # outer_f = np.zeros([96, 96])
        # outer = output_test[2] * 255
        # outer_f[outer>=0.5]=1
        #
        #
        # fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        # fig.tight_layout()
        # # ax.axis('off')
        #
        # ax[0,0].imshow(inp, cmap='gray')
        # ax[0,1].imshow(inner_f, cmap='gray')
        #
        # ax[1,0].imshow(inp, cmap='gray')
        # ax[1, 1].imshow(outer_f, cmap='gray')
        # path = batch["path"][0].split('/')[5] + '+' + batch["path"][0].split('/')[6] + '+' + \
        #        batch["path"][0].split('/')[9]
        # # print(path)
        #
        # plt.savefig("demo_prob/{}.png".format(path))
        # plt.close("all")
        # sys.exit()
        # if not self.training:
        #     return output_f

        output_f = self.gcn(output_f, cnn_feature,  batch)
        # output_f = self.gcn(output_f, cnn_feature, batch)
        # print("output_shape:",np.array(output_f["py_pred"]).shape)
        # print("output_shape:", np.array(output_f["py_pred"])[-1].shape)
        # output_shape: torch.Size([6, 128, 2])
        # cnn_feature_size: torch.Size([3, 64, 96, 96])
        # sys.exit()
        return output_f

    # Bayesian Uncertainty ellipse
    # def forward(self, x, batch=None):
    #     # output, cnn_feature = self.dla(x)
    #     # print("input_shape:",x.shape)
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # sys.exit()
    #     output_f = {}
    #     x = torch.unsqueeze(x, 1)
    #     # output, cnn_feature = self.hybrid(x)
    #     # prob_health_t,_,seg_from_cls,_ = self.hybrid_cls(x)
    #     prob_health_t,plq_seg_from_cls = self.hybrid_cls(x)
    #     prob_health = prob_health_t.detach()
    #     # prob_health = self.hybrid_cls(x)[:,7,:]
    #     # print("prob_shape:",prob_health.shape)
    #     # sys.exit()
    #     # print("before hybrid1")
    #
    #     output_h, cnn_feature_h,output_h_seg,cnn_feature_h_seg = self.hybrid_1(x)
    #
    #     # print("cnn_feature_h:",cnn_feature_h.shape)
    #     # output_h, cnn_feature_h,output_seg_h_0,output_seg_h_1,cnn_feature_h_seg = self.hybrid_1(x)
    #
    #     output_unh, cnn_feature_unh,output_seg_unh,cnn_feature_unh_seg = self.hybrid_2(x)
    #
    #     # print("F.softmax(prob_health)[:, 0]:",F.softmax(prob_health)[:, 0].shape)
    #     # print("F.softmax(prob_health)[:, 0].reshape(-1, 1, 1, 1):", F.softmax(prob_health)[:, 0].reshape(-1, 1, 1, 1).shape)
    #     # sys.exit()
    #     output = torch.mul(F.softmax(prob_health)[:, 0].reshape(-1, 1, 1, 1).expand(-1, 32, 96, 96),
    #                        cnn_feature_h) \
    #              + torch.mul(F.softmax(prob_health)[:, 1].reshape(-1, 1, 1, 1).expand(-1, 32, 96, 96),
    #                          cnn_feature_h) \
    #              + torch.mul(F.softmax(prob_health)[:, 2].reshape(-1, 1, 1, 1).expand(-1, 32, 96, 96),
    #                          cnn_feature_unh)
    #     output_seg = torch.mul(F.softmax(prob_health)[:, 0].reshape(-1, 1, 1, 1).expand(-1, 32, 96, 96),
    #                         cnn_feature_h_seg) \
    #                 + torch.mul(F.softmax(prob_health)[:, 1].reshape(-1, 1, 1, 1).expand(-1, 32, 96, 96),
    #                             cnn_feature_h_seg) \
    #                 + torch.mul(F.softmax(prob_health)[:, 2].reshape(-1, 1, 1, 1).expand(-1, 32, 96, 96),
    #                             cnn_feature_unh_seg)
    #
    #     # output_seg =  torch.mul(F.softmax(prob_health)[:, 2].reshape(-1, 1, 1, 1).expand(-1, 32, 96, 96),
    #     #                         cnn_feature_unh_seg)
    #
    #
    #     output,cnn_feature = self.hybrid_block(output)
    #     output_seg, cnn_feature_seg = self.hybrid_block_seg(output_seg)
    #
    #
    #
    #
    #     output_f["prob_health"] =F.softmax(prob_health_t, 1)
    #     # output_f["seg_from_cls"] =F.softmax( seg_from_cls,1)
    #     output_f["prob_map"] = output
    #     output_f["prob_map_seg"] =F.softmax(output_seg,1)
    #
    #
    #
    #     output_f = self.gcn(output_f, cnn_feature,cnn_feature_seg, batch)
    #     # output_f = self.gcn(output_f, cnn_feature, batch)
    #     # print("output_shape:",np.array(output_f["py_pred"]).shape)
    #     # print("output_shape:", np.array(output_f["py_pred"])[-1].shape)
    #     # output_shape: torch.Size([6, 128, 2])
    #     # cnn_feature_size: torch.Size([3, 64, 96, 96])
    #     # sys.exit()
    #     return output_f





    # original classfier+2hybridresunet
    # def forward(self, x, batch=None):
    #     # output, cnn_feature = self.dla(x)
    #     # print("input_shape:",x.shape)
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # sys.exit()
    #     x = torch.unsqueeze(x, 1)
    #     # output, cnn_feature = self.hybrid(x)
    #     prob_health = self.hybrid_cls(x)
    #     output_health,  cnn_feature_h = self.hybrid_1(x)
    #     output_unhealth,cnn_feature_unh = self.hybrid_2(x)
    #     output_f = {}
    #     output_f["prob_health"] =F.softmax(prob_health)
    #     output_f["output_health"] = output_health
    #     output_f["output_unhealth"] = output_unhealth
    #     # print("prob_health[:,0]:",prob_health[:,0].reshape(-1,1,1,1).shape)
    #     # print("output_health:", output_health.shape)
    #     # print("cnn_feature_h:", cnn_feature_h.shape)
    #     # print("prob_health[:,0].expand(-1,64, 96, 96):",F.softmax(prob_health)[:,0].reshape(-1,1,1,1).expand(-1,2, 2, 2))
    #     # print("prob_health[:,0].expand(-1,64, 96, 96):", F.softmax(prob_health)[:, 1].reshape(-1, 1, 1, 1).expand(-1, 3, 2, 2))
    #     # sys.exit()
    #
    #     # output_f["prob_map"] = torch.mul(F.softmax(prob_health)[:,0].reshape(-1,1,1,1).expand(-1,3, 96, 96),output_health) \
    #     #                        +torch.mul(F.softmax(prob_health)[:,1].reshape(-1,1,1,1).expand(-1,3, 96, 96),output_unhealth)\
    #     #                        +torch.mul(F.softmax(prob_health)[:,2].reshape(-1,1,1,1).expand(-1,3, 96, 96),output_unhealth)
    #     # cnn_feature = torch.mul(F.softmax(prob_health)[:,0].reshape(-1,1,1,1).expand(-1,64, 96, 96),cnn_feature_h)\
    #     #               +torch.mul(F.softmax(prob_health)[:,1].reshape(-1,1,1,1).expand(-1,64, 96, 96),cnn_feature_unh)\
    #     #               +torch.mul(F.softmax(prob_health)[:,2].reshape(-1,1,1,1).expand(-1,64, 96, 96),cnn_feature_unh)
    #
    #     output_f["prob_map"] = torch.mul(F.softmax(prob_health)[:, 0].reshape(-1, 1, 1, 1).expand(-1, 3, 96, 96),
    #                                      output_health) \
    #                            + torch.mul(F.softmax(prob_health)[:, 1].reshape(-1, 1, 1, 1).expand(-1, 3, 96, 96),
    #                                        output_health) \
    #                            + torch.mul(F.softmax(prob_health)[:, 2].reshape(-1, 1, 1, 1).expand(-1, 3, 96, 96),
    #                                        output_unhealth)
    #     cnn_feature = torch.mul(F.softmax(prob_health)[:, 0].reshape(-1, 1, 1, 1).expand(-1, 64, 96, 96), cnn_feature_h) \
    #                   + torch.mul(F.softmax(prob_health)[:, 1].reshape(-1, 1, 1, 1).expand(-1, 64, 96, 96),
    #                               cnn_feature_h) \
    #                   + torch.mul(F.softmax(prob_health)[:, 2].reshape(-1, 1, 1, 1).expand(-1, 64, 96, 96),
    #                               cnn_feature_unh)
    #     # cnn_feature = prob_health[:0]*cnn_feature_h +prob_health[:1]*cnn_feature_unh
    #     # print("output_size:",output.shape)
    #     # print("cnn_feature_size:", cnn_feature.shape)
    #     # print("mask_shape:",batch["mask"].shape)
    #     # mask_shape: torch.Size([3, 192, 192])
    #     #
    #     # # sys.exit()
    #     # inp = img_utils.unnormalize_img(batch['inp'][0][15 // 2, :, :], mean[:, :, 15 // 2], std[:, :, 15 // 2])
    #     # output_test= F.softmax(output_f["prob_map"],dim=1)[-1].cpu().numpy()
    #     # inner=output_test[1]*255
    #     # inner_f=np.zeros([96,96])
    #     # inner_f[inner>=0.5]=1
    #     # outer_f = np.zeros([96, 96])
    #     # outer = output_test[2] * 255
    #     # outer_f[outer>=0.5]=1
    #     #
    #     #
    #     # fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    #     # fig.tight_layout()
    #     # # ax.axis('off')
    #     #
    #     # ax[0,0].imshow(inp, cmap='gray')
    #     # ax[0,1].imshow(inner_f, cmap='gray')
    #     #
    #     # ax[1,0].imshow(inp, cmap='gray')
    #     # ax[1, 1].imshow(outer_f, cmap='gray')
    #     # path = batch["path"][0].split('/')[5] + '+' + batch["path"][0].split('/')[6] + '+' + \
    #     #        batch["path"][0].split('/')[9]
    #     # # print(path)
    #     #
    #     # plt.savefig("demo_prob/{}.png".format(path))
    #     # plt.close("all")
    #     # sys.exit()
    #
    #     output_f = self.gcn(output_f, cnn_feature,  batch)
    #     # output_f = self.gcn(output_f, cnn_feature, batch)
    #     # print("output_shape:",np.array(output_f["py_pred"]).shape)
    #     # print("output_shape:", np.array(output_f["py_pred"])[-1].shape)
    #     # output_shape: torch.Size([6, 128, 2])
    #     # cnn_feature_size: torch.Size([3, 64, 96, 96])
    #     # sys.exit()
    #     return output_f

    #
    #Time_seties


    # def forward(self, x, batch=None):
    #     # output, cnn_feature = self.dla(x)
    #     # print("input_shape:",x.shape)
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # sys.exit()
    #     # x=torch.unsqueeze(x,1)
    #     output, cnn_feature,out_clstm = self.Timeseries(x)
    #     # output, cnn_feature,output_seg,cnn_feature_seg = self.hybrid_2(x)
    #     output_f={}
    #     output_f["prob_map"]=output
    #     output_f["out_clstm"] = out_clstm
    #     # output_f["prob_map_seg"] = output_seg
    #     # print("output_size:",output.shape)
    #     # print("cnn_feature_size:", cnn_feature.shape)
    #     # print("mask_shape:",batch["mask"].shape)
    #     # mask_shape: torch.Size([3, 192, 192])
    #     #
    #     # # sys.exit()
    #     # inp = img_utils.unnormalize_img(batch['inp'][0][15 // 2, :, :], mean[:, :, 15 // 2], std[:, :, 15 // 2])
    #     # output_test= F.softmax(output_f["prob_map"],dim=1)[-1].cpu().numpy()
    #     # inner=output_test[1]*255
    #     # inner_f=np.zeros([96,96])
    #     # inner_f[inner>=0.5]=1
    #     # outer_f = np.zeros([96, 96])
    #     # outer = output_test[2] * 255
    #     # outer_f[outer>=0.5]=1
    #     #
    #     #
    #     # fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    #     # fig.tight_layout()
    #     # # ax.axis('off')
    #     #
    #     # ax[0,0].imshow(inp, cmap='gray')
    #     # ax[0,1].imshow(inner_f, cmap='gray')
    #     #
    #     # ax[1,0].imshow(inp, cmap='gray')
    #     # ax[1, 1].imshow(outer_f, cmap='gray')
    #     # path = batch["path"][0].split('/')[5] + '+' + batch["path"][0].split('/')[6] + '+' + \
    #     #        batch["path"][0].split('/')[9]
    #     # # print(path)
    #     #
    #     # plt.savefig("demo_prob/{}.png".format(path))
    #     # plt.close("all")
    #     # sys.exit()
    #
    #     # output_f = self.gcn(output_f, cnn_feature,cnn_feature_seg, batch)
    #     # print("before gcn")
    #     output_f = self.gcn(output_f, cnn_feature, batch)
    #     # print("output_shape:",np.array(output_f["py_pred"]).shape)
    #     # print("output_shape:", np.array(output_f["py_pred"])[-1].shape)
    #     # output_shape: torch.Size([6, 128, 2])
    #     # cnn_feature_size: torch.Size([3, 64, 96, 96])
    #     # sys.exit()
    #     return output_f

    # force inside
    # def forward(self, x, batch=None):
    #     # output, cnn_feature = self.dla(x)
    #     # print("input_shape:",x.shape)
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # input_shape: torch.Size([18, 15, 192, 192])
    #     # sys.exit()
    #     x=torch.unsqueeze(x,1)
    #     # output, cnn_feature = self.hybrid_1(x)
    #     output, cnn_feature,output_seg_0,output_seg_1,cnn_feature_seg = self.hybrid_2(x)
    #     output_f={}
    #     output_seg_0 = F.softmax(output_seg_0, 1)
    #     output_seg_1_temp= F.softmax(output_seg_1, 1)
    #     output_f["prob_map"]=output
    #     # output_f["out_clstm"] = out_clstm
    #     output_seg_1 = output_seg_1_temp.clone()
    #     output_seg_1[:,1,:,:] = output_seg_0[:,1,:,:]*output_seg_1_temp[:,1,:,:]
    #     # print("before output_seg_1_shape:",output_seg_1.shape)
    #
    #     # output_seg_1 = output_seg_1/output_seg_1.sum(dim = 1).unsqueeze(1)
    #     # print("after output_seg_1_shape:",output_seg_1.shape)
    #
    #     # print("before torch.unsqueeze")
    #
    #     output_f["prob_map_seg_0"] =output_seg_0
    #     output_f["prob_map_seg_1"] = output_seg_1
    #
    #     # print("output_size:",output.shape)
    #     # print("cnn_feature_size:", cnn_feature.shape)
    #     # print("mask_shape:",batch["mask"].shape)
    #     # mask_shape: torch.Size([3, 192, 192])
    #     #
    #     # # sys.exit()
    #     # inp = img_utils.unnormalize_img(batch['inp'][0][15 // 2, :, :], mean[:, :, 15 // 2], std[:, :, 15 // 2])
    #     # output_test= F.softmax(output_f["prob_map"],dim=1)[-1].cpu().numpy()
    #     # inner=output_test[1]*255
    #     # inner_f=np.zeros([96,96])
    #     # inner_f[inner>=0.5]=1
    #     # outer_f = np.zeros([96, 96])
    #     # outer = output_test[2] * 255
    #     # outer_f[outer>=0.5]=1
    #     #
    #     #
    #     # fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    #     # fig.tight_layout()
    #     # # ax.axis('off')
    #     #
    #     # ax[0,0].imshow(inp, cmap='gray')
    #     # ax[0,1].imshow(inner_f, cmap='gray')
    #     #
    #     # ax[1,0].imshow(inp, cmap='gray')
    #     # ax[1, 1].imshow(outer_f, cmap='gray')
    #     # path = batch["path"][0].split('/')[5] + '+' + batch["path"][0].split('/')[6] + '+' + \
    #     #        batch["path"][0].split('/')[9]
    #     # # print(path)
    #     #
    #     # plt.savefig("demo_prob/{}.png".format(path))
    #     # plt.close("all")
    #     # sys.exit()
    #
    #     output_f = self.gcn(output_f, cnn_feature,cnn_feature_seg, batch)
    #     # output_f = self.gcn(output_f, cnn_feature, batch)
    #     # print("output_shape:",np.array(output_f["py_pred"]).shape)
    #     # print("output_shape:", np.array(output_f["py_pred"])[-1].shape)
    #     # output_shape: torch.Size([6, 128, 2])
    #     # cnn_feature_size: torch.Size([3, 64, 96, 96])
    #     # sys.exit()
    #     return output_f


def get_network(num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
    network = Network(num_layers, heads, head_conv, down_ratio, det_dir)
    return network
