import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import net_utils
import torch
import sys
from .loss import *
class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.hybrid_crit=WeightedHausdorffDistanceDoubleBoundLoss()
        self.hybrid_crit_seg =GeneralizedDiceLoss()
        self.l2 = nn.MSELoss()
        self.BCE = nn.BCELoss(weight=torch.Tensor([1,2,5]))

        self.ct_crit = net_utils.FocalLoss()
        self.wh_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.reg_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.ex_crit = torch.nn.functional.smooth_l1_loss
        self.py_crit = torch.nn.functional.smooth_l1_loss
        self.clstm_loss = torch.nn.functional.smooth_l1_loss
    # def forward(self, batch):
    #     # print("batch:",batch)
    #     output = self.net(batch['inp'], batch)
    #
    #     scalar_stats = {}
    #     loss = 0
    #     # print("output['prob_map']_shape",output['prob_map'].shape)
    #     # print("batch['mask']_shape", batch['mask'].shape)
    #     # sys.exit()
    #     # sys.exit()
    #     # output['prob_map']_shape torch.Size([3, 3, 192, 192])
    #     # batch['mask']_shape torch.Size([3, 192, 192])
    #     risk_label = batch['risk_label']
    #     risk_label = risk_label.type(torch.cuda.FloatTensor)
    #
    #     risk_loss = self.l2(F.softmax(output['prob'],dim=1), risk_label)
    #     scalar_stats.update({'risk_loss':risk_loss})
    #     loss += risk_loss
    #
    #
    #     scalar_stats.update({'loss': loss})
    #     image_stats = {}
    #
    #     return output, loss, scalar_stats, image_stats
    def forward(self, batch,epoch):
        # print("batch:",batch)


        output = self.net(batch['inp'], batch)
        # if epoch <= 0:
        #     scalar_stats = {}
        #     loss = 0
        #     # print("output['prob_map']_shape",output['prob_map'].shape)
        #     # print("batch['mask']_shape", batch['mask'].shape)
        #     # sys.exit()
        #     # sys.exit()
        #     # output['prob_map']_shape torch.Size([3, 3, 192, 192])
        #     # batch['mask']_shape torch.Size([3, 192, 192])
        #     # hybrid_health_loss = self.hybrid_crit(F.softmax(output['output_health'], dim=1), batch['mask'])
        #     # scalar_stats.update({'hybrid_health_loss': hybrid_health_loss})
        #     # loss += hybrid_health_loss
        #     # mask_seg = batch['mask_seg_plaque'].type(torch.cuda.LongTensor)
        #     # hybrid_loss_seg = self.hybrid_crit_seg(F.softmax(output['prob_map_seg'], dim=1), mask_seg)
        #     hybrid_loss = self.hybrid_crit(F.softmax(output['prob_map'], dim=1), batch['mask'])
        #     scalar_stats.update({'hybrid_loss': hybrid_loss})
        #
        #     label_health = batch['label_health'].type(torch.cuda.FloatTensor)
        #     class_health_loss = self.BCE(output['prob_health'], label_health)
        #     scalar_stats.update({'Class_health_loss': class_health_loss})
        #     loss += class_health_loss
        #     scalar_stats.update({'loss': loss})
        #     image_stats = {}
        #
        #     return output, loss, scalar_stats, image_stats



        scalar_stats = {}
        loss = 0
        # print("output['prob_map']_shape",output['prob_map'].shape)
        # print("batch['mask']_shape", batch['mask'].shape)
        # sys.exit()
        # sys.exit()
        # output['prob_map']_shape torch.Size([3, 3, 192, 192])
        # batch['mask']_shape torch.Size([3, 192, 192])

        # print("output['prob_map']:",output['prob_map'])
        # print("output['prob_map'].shape:",output['prob_map'].shape)
        # print("batch['mask]:",batch['mask'].shape)
        #
        # mask = batch['mask'].detach().cpu().numpy()
        # print("mask_unique:",np.unique(mask))

        # print('batch_unique:',np.unique(batch['mask'].cpu()))

        hybrid_loss = self.hybrid_crit(F.softmax(output['prob_map'],dim=1), batch['mask'])
        scalar_stats.update({'hybrid_loss':hybrid_loss})
        loss += hybrid_loss

# #clstm start
#         clstm_loss = 0
#
#         output_clstm = torch.squeeze(output['out_clstm'],2)
#
#
#         # for i in range(output['out_clstm'].size(1)-1):
#         #     # # print("output_clstm[:,i,:,:]:",output_clstm[:,i,:,:].shape)
#         #     # print("output['out_clstm']:",output['out_clstm'].shape)
#         #     # print("batch['inp']:",batch['inp'].shape)
#         #     clstm_loss += self.clstm_loss(output_clstm[:,i,:,:], batch['inp'][:,i+1,:,:])
#         #
#         #
#         #
#         # scalar_stats.update({'clstm_loss':clstm_loss})
#         # loss += 0.01*clstm_loss
# #clstm end
#ellipse
        # hybrid_health_loss = self.hybrid_crit(F.softmax(output['output_health'], dim=1), batch['mask'])
        # scalar_stats.update({'hybrid_health_loss': hybrid_health_loss})
        # loss += hybrid_health_loss
        # mask_seg_plq_only_fromcls = batch['mask_seg_plq_only'].type(torch.cuda.LongTensor)
        # hybrid_loss_seg_plq_only_fromcls = self.hybrid_crit_seg(output['plq_seg_from_cls'], mask_seg_plq_only_fromcls)
        # scalar_stats.update({'hybrid_loss_seg_plq_only_fromcls': hybrid_loss_seg_plq_only_fromcls})
        # loss += hybrid_loss_seg_plq_only_fromcls

        mask_seg_wall = batch['mask_seg_wall'].type(torch.cuda.LongTensor)
        hybrid_loss_seg_wall = self.hybrid_crit_seg(output['prob_map_seg'], mask_seg_wall)
        scalar_stats.update({'hybrid_loss_seg_wall': hybrid_loss_seg_wall})
        loss += hybrid_loss_seg_wall

        mask_seg_wall_cls = batch['mask_seg_wall'].type(torch.cuda.LongTensor)
        hybrid_loss_seg_wall_cls = self.hybrid_crit_seg(output['seg_from_cls'], mask_seg_wall_cls)
        scalar_stats.update({'hybrid_loss_seg_wall_from_cls': hybrid_loss_seg_wall_cls})
        loss +=hybrid_loss_seg_wall_cls


        # mask_seg_plq_only = batch['mask_seg_plq_only'].type(torch.cuda.LongTensor)
        # check = torch.sum(mask_seg_plq_only)
        # print("check_outLoss:",check)
        # hybrid_loss_seg_plq_only = self.hybrid_crit_seg(output['prob_map_seg_1'], mask_seg_plq_only)



        # scalar_stats.update({'hybrid_loss_seg_plq_only': hybrid_loss_seg_plq_only})
        # loss += hybrid_loss_seg_plq_only

        #
        label_health = batch['label_health'].type(torch.cuda.FloatTensor)
        class_health_loss = self.BCE(output['prob_health'],label_health)
        scalar_stats.update({'Class_health_loss': class_health_loss})
        # if epoch <= 6:
        loss += 10*class_health_loss
#ellipse



        # print("epoch known:")

        # mask_dis = batch['mask_dis'].type(torch.cuda.FloatTensor)
        # hybrid_loss_seg = self.l2(output['prob_map_seg'],mask_dis)
        # scalar_stats.update({'hybrid_loss_seg': hybrid_loss_seg})
        # loss += 5*hybrid_loss_seg

        # wh_loss = self.wh_crit(output['wh'], batch['wh'], batch['ct_ind'], batch['ct_01'])
        # scalar_stats.update({'wh_loss': wh_loss})
        # loss += 0.1 * wh_loss c

        # reg_loss = self.reg_crit(output['reg'], batch['reg'], batch['ct_ind'], batch['ct_01'])c
        # scalar_stats.update({'reg_loss': reg_loss})
        # loss += reg_loss
        # print("i_gt_4py.shape:",output['i_gt_4py'].shape)
        # sys.exit()
        # ex_loss = self.ex_crit(output['ex_pred'], output['i_gt_4py'])
        # scalar_stats.update({'ex_loss': ex_loss})
        # loss += ex_loss
        if not self.training:
            scalar_stats.update({'loss': loss})
            image_stats = {}

            return output, loss, scalar_stats, image_stats


        py_loss = 0
        output['py_pred'] = [output['py_pred'][-1]]
        for i in range(len(output['py_pred'])):
            py_loss += self.py_crit(output['py_pred'][i], output['i_gt_py']) / len(output['py_pred'])
        scalar_stats.update({'py_loss': py_loss})
        loss += py_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

