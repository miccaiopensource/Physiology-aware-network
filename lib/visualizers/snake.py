from lib.utils import img_utils, data_utils
from lib.utils.snake import snake_config
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import cycle
import os
from termcolor import colored
import SimpleITK as sitk
import cv2

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
from medpy.metric.binary import hd95, asd, hd
import sys


class Visualizer:
    def __init__(self):
        self.hausdorff = [[] for _ in range(2)]
        self.hausdorff_healthy_cal = [[] for _ in range(2)]
        self.hausdorff_unhealthy_cal = [[] for _ in range(2)]
        self.hausdorff_healthy_noncal = [[] for _ in range(2)]
        self.hausdorff_unhealthy_noncal = [[] for _ in range(2)]
        self.sort_hdf = [[] for _ in range(2)]
        self.p_involve = [[] for _ in range(2)]
        self.aveiou = []
        self.aveiou_inner = []
        self.aveiou_outer = []
        self.avedice = []
        self.avedice_inner = []
        self.avedice_outer = []
        self.avehausdorff = []
        self.avehausdorff_healthy_cal = []
        self.avehausdorff_unhealthy_cal = []
        self.avehausdorff_healthy_noncal = []
        self.avehausdorff_unhealthy_noncal = []
        self.avehausdorff_int = [[] for _ in range(2)]
        self.class_ct_inner = [0,0,0]
        self.class_ct_outer = [0, 0, 0]
        self.class_ct = [0, 0, 0]
        self.ill_amount_inner = 0
        self.ill_amount_outer = 0
        self.ill_amount = 0

    def visualize_ex(self, output, batch):
        # print("batch['inp'][0][15//2,:,:]:",batch['inp'][0][15//2,:,:].shape)
        inp = img_utils.bgr_to_rgb(
            img_utils.unnormalize_img(batch['inp'][0][15 // 2, :, :], mean[:, :, 15 // 2], std[:, :, 15 // 2]).permute(
                1, 2, 0))
        ex = output['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy() * snake_config.down_ratio

        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)

        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        for i in range(len(ex)):
            color = next(colors).tolist()
            poly = ex[i]
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color)

        plt.savefig("demo.png")
    def iou(self, pred, target, n_classes=2):
        ious = []

    # Ignore IoU for background class ("0")
        for cls in range(1, n_classes):
            pred_inds = pred == cls
            target_inds = target == cls

        # Compute intersection
            intersection = (pred_inds & target_inds).sum().item()
            print("intersection:",intersection)

        # Compute union
            union = pred_inds.sum().item() + target_inds.sum().item() - intersection
            print("union:",union)

            if union == 0:
                ious.append(float('nan'))
            else:
                ious.append(float(intersection) / float(max(union, 1)))

        return np.array(ious)
    def dice(self,pred, target, n_classes=2):
        dices = []
        for cls in range(1, n_classes):
            pred_inds = pred == cls
            target_inds = target == cls
            intersection =2 * (pred_inds & target_inds).sum().item()
            print("intersection:",intersection)
            union = pred_inds.sum().item() + target_inds.sum().item()
            print("union:",union)
            if union == 0:
                dices.append(float('nan'))
            else:
                dices.append(float(intersection) / float(max(union, 1)))

        return np.array(dices)
    def fill(self,data, start_coords, fill_value):
        """
        Flood fill algorithm

        Parameters
        ----------
        data : (M, N) ndarray of uint8 type
            Image with flood to be filled. Modified inplace.
        start_coords : tuple
            Length-2 tuple of ints defining (row, col) start coordinates.
        fill_value : int
            Value the flooded area will take after the fill.

        Returns
        -------
        None, ``data`` is modified inplace.
        """
        xsize, ysize = data.shape
        orig_value = data[start_coords[0], start_coords[1]]

        stack = set(((start_coords[0], start_coords[1]),))
        if fill_value == orig_value:
            raise ValueError("Filling region with same value "
                             "already present is unsupported. "
                             "Did you already fill this region?")
        while stack:
            x, y = stack.pop()
            if data[x, y] == orig_value:
                data[x, y] = fill_value
                if x > 0:
                    stack.add((x - 1, y))
                if x < (xsize - 1):
                    stack.add((x + 1, y))
                if y > 0:
                    stack.add((x, y - 1))
                if y < (ysize - 1):
                    stack.add((x, y + 1))

    def visualize_training_box(self, output, batch, id):
        from os.path import exists
        from os import mkdir
        from os.path import join
        from lib.config import cfg, args
        from lib.utils.snake.snake_gcn_utils import uniform_upsample

        inp = img_utils.unnormalize_img(batch['inp'][0][15 // 2, :, :], mean[:, :, 15 // 2], std[:, :, 15 // 2])

        if 'py' in output.keys():
            temp = output['py'][-1].unsqueeze(0)
        else:
            return 0,0,0,0

        py_p = uniform_upsample(temp, 960)
        py_p = py_p.squeeze(0)

        py_p = py_p.detach().cpu().numpy() * snake_config.down_ratio
        cls = output['cls']
        mask_b = np.zeros([2,96, 96])
        ex = output['py']
        cls = output['cls']
        print("cls_vis:", cls)
        mask_check = batch['mask'][0].cpu().numpy()

        gt = batch['i_gt_py'][-1].detach().cpu().numpy() * snake_config.down_ratio
        gt_m = batch["mask"][-1].detach().cpu().numpy()
        gt_seg = batch["mask_seg"][-1].detach().cpu().numpy()
        output_test = F.softmax(output["prob_map"], dim=1)[-1].cpu().numpy()
        if "prob_map_seg" in output.keys():
            output_seg_wall = output["prob_map_seg"][-1].cpu().numpy()
        print("prob_test_shape:",output_test.shape)

        # for BCL
        prob_test_sum = np.sum(output_test,axis =-1)
        prob_test_sum = np.sum(prob_test_sum,axis =-1)
        print("prob_test_sum:",prob_test_sum)
        mask = np.zeros(output_test.shape)
        if "prob_map_seg" in output.keys():
            mask_seg_wall = output_seg_wall[1]


        mask[1][output_test[1] > snake_config.threshold] = 255
        mask[2][output_test[2] > snake_config.threshold] = 255
        inner = mask[1]
        outer = mask[2]

        h, w = gt_seg.shape[:]
        gt_seg_wall = np.zeros([h,w])
        gt_seg_wall[gt_seg == 2] = 1
        gt_seg_wall[gt_seg == 3] = 1
        gt_seg_wall[gt_seg == 4] = 1
        gt_seg_lumen = np.zeros([h,w])
        gt_seg_lumen[gt_seg == 1] = 1
        gt_seg_artery = np.zeros([h,w])
        gt_seg_artery[gt_seg == 1] = 1
        gt_seg_artery[gt_seg == 2] = 1
        gt_seg_artery[gt_seg == 3] = 1
        gt_seg_artery[gt_seg == 4] = 1


        gt_seg_plq = np.zeros([2,h,w])
        gt_seg_plq[0][gt_seg ==  3]  = 1
        gt_seg_plq[1][gt_seg == 4] = 1
        gt_seg_rgb = np.zeros([h,w,3])

        gt_seg_rgb[gt_seg == 1] = [0, 0, 255]
        gt_seg_rgb[gt_seg == 2] = [255, 255, 255]
        gt_seg_rgb[gt_seg == 3] = [255, 128, 0]
        gt_seg_rgb[gt_seg == 4] = [255, 0, 0]
        ct_cls = np.unique(gt_seg)
        p_involve_in = 1
        p_involve_out = 1

        lambda_nc = 10
        lambda_c = 5
        e = 0.0001
        if 3 in ct_cls or 4 in ct_cls:
            if 0 in cls:
                sum_c = np.sum(gt_seg_plq[0])
                right_c = np.sum(np.multiply(gt_seg_plq[0], mask_b[0]))
                sum_nc = np.sum(gt_seg_plq[1])
                right_nc = np.sum(np.multiply(gt_seg_plq[1], mask_b[0]))
                p_involve_in = np.maximum(float((lambda_c-1) *(1-(right_c+e)/(sum_c+e))+1),
                                          float((lambda_nc-1) *(1-(right_nc+e)/(sum_nc+e))+1))

            if 1 in cls:
                sum_c = np.sum(gt_seg_plq[0])
                wrong_c = np.sum(np.multiply(gt_seg_plq[0], mask_b[1]))
                sum_nc = np.sum(gt_seg_plq[1])
                wrong_nc = np.sum(np.multiply(gt_seg_plq[1], mask_b[1]))
                p_involve_out = np.maximum(float((lambda_c - 1) * (1 - (sum_c-wrong_c + e) / (sum_c + e)) + 1),
                                          float((lambda_nc - 1) * (1 - (sum_nc-wrong_nc + e) / (sum_nc + e)) + 1))
        self.p_involve[0].append(p_involve_in)
        self.p_involve[1].append(p_involve_out)


        temp = output['py'][-1].unsqueeze(0)
        py = uniform_upsample(temp, 960)
        py = py.squeeze(0)
        py = py.detach().cpu().numpy() * snake_config.down_ratio

        gt_py = batch['i_gt_py'][-1].detach().cpu().numpy() * snake_config.down_ratio
        i_it_py = batch['i_it_py'][-1].detach().cpu().numpy() * snake_config.down_ratio
        evo_i_it_py = output['i_it_py'].detach().cpu().numpy() * snake_config.down_ratio
        if len(py) == 0:
            return
        ct_cls = batch['ct_cls'][-1].detach().cpu().numpy()


        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        py_mask = np.zeros([2, 96, 96])
        py_mask_tmp= np.zeros([2,96,96])
        gt_py_mask_f = np.zeros([2, 96, 96])

        gt_py_mask = np.zeros([2, 96, 96])
        for i in range(2):
            gt_py_mask_f[i][gt_m == i + 1] = 1

            cv2.imwrite('mask_gt_py_f_{}_m.png'.format(i), gt_py_mask_f[i] * 255)

        isOutIndex = False
        for i in range(len(py)):
            for ord in py[i]:
                if ord[0] < 0:
                    ord[0] = 0
                    isOutIndex = True

                if ord[0] > 95:
                    ord[0] = 95
                    isOutIndex = True
                if ord[1] < 0:
                    ord[1] = 0
                    isOutIndex = True
                if ord[1] > 95:
                    ord[1] = 95
                    isOutIndex = True
                py_mask[int(cls[i])][int(ord[1])][int(ord[0])] = 1
                py_mask_tmp[int(cls[i])][int(ord[1])][int(ord[0])] = 1
        if isOutIndex:
            print("isOutIndex")
            return self.avehausdorff_int,0,0,0
        self.fill(py_mask_tmp[0],(0,0),1)
        self.fill(py_mask_tmp[1],(0,0),1)
        py_seg_wall = py_mask_tmp[0] - py_mask_tmp[1]
        py_seg_lumen =1- py_mask_tmp[0]
        py_seg_artery = 1-py_mask_tmp[1]

        hdf = self.hausdorff_distance(py_mask, gt_py_mask_f)
        self.hausdorff[0].append(hdf[0])
        self.hausdorff[1].append(hdf[1])
        ct_cls = np.unique(gt_seg)
        if 4 in ct_cls or 3 in ct_cls:
            self.hausdorff_unhealthy_cal[0].append(hdf[0])
            self.hausdorff_unhealthy_cal[1].append(hdf[1])
        else:
            self.hausdorff_healthy_cal[0].append(hdf[0])
            self.hausdorff_healthy_cal[1].append(hdf[1])
        if 4 in ct_cls :
            self.hausdorff_unhealthy_noncal[0].append(hdf[0])
            self.hausdorff_unhealthy_noncal[1].append(hdf[1])
        else:
            self.hausdorff_healthy_noncal[0].append(hdf[0])
            self.hausdorff_healthy_noncal[1].append(hdf[1])

        path = batch["path"][0].split('/')[5] + '+' + batch["path"][0].split('/')[6] + '+' + \
               batch["path"][0].split('/')[9]
        self.sort_hdf[0].append((float(hdf[0]+hdf[1])/2))
        self.sort_hdf[1].append(path)
        hdf_array_sort_index = np.argsort(np.array(self.sort_hdf[0]))


        ave_hdf_0_healthy_cal = np.average(self.hausdorff_healthy_cal[0])
        ave_hdf_1_healthy_cal = np.average(self.hausdorff_healthy_cal[1])
        ave_hdf_0_unhealthy_cal = np.average(self.hausdorff_unhealthy_cal[0])
        ave_hdf_1_unhealthy_cal = np.average(self.hausdorff_unhealthy_cal[1])
        ave_hdf_0_healthy_noncal = np.average(self.hausdorff_healthy_noncal[0])
        ave_hdf_1_healthy_noncal = np.average(self.hausdorff_healthy_noncal[1])
        ave_hdf_0_unhealthy_noncal = np.average(self.hausdorff_unhealthy_noncal[0])
        ave_hdf_1_unhealthy_noncal = np.average(self.hausdorff_unhealthy_noncal[1])
        ave_hdf_0 = np.average(self.hausdorff[0])
        ave_hdf_1 = np.average(self.hausdorff[1])
        ave_p_involve_in = np.average(self.p_involve[0])
        ave_p_involve_out = np.average(self.p_involve[1])
        print("hdf_0:", hdf[0], "hdf_1:", hdf[1])
        print("ave_hdf_0:", ave_hdf_0, "ave_hdf_1:", ave_hdf_1)
        print("ave_hdf_0_unhealthy_cal:", ave_hdf_0_unhealthy_cal, "ave_hdf_1_unhealthy_cal:", ave_hdf_1_unhealthy_cal)
        print("ave_hdf_0_unhealthy_noncal:", ave_hdf_0_unhealthy_noncal, "ave_hdf_1_unhealthy_noncal:", ave_hdf_1_unhealthy_noncal)
        self.avehausdorff_int[0] = list(map(int, self.hausdorff[0]))
        self.avehausdorff_int[1] = list(map(int, self.hausdorff[1]))
#iou
        iou = self.iou(py_seg_wall, gt_seg_wall)
        print("iou:", iou)
        iou_lumen = self.iou(py_seg_lumen, gt_seg_lumen)
        print("iou_lumen:", iou_lumen)
        iou_artery = self.iou(py_seg_artery, gt_seg_artery)
        print("iou_artery:", iou_artery)
        path = batch["path"][0].split('/')[5] + '+' + batch["path"][0].split('/')[6] + '+' + \
               batch["path"][0].split('/')[9]

        iou0check = False
        if(iou_artery>0):
            self.aveiou.append(iou)
            self.aveiou_inner.append(iou_lumen)
            self.aveiou_outer.append(iou_artery)
        else:
            iou0check = True
            np.savetxt('metric_seg/py_seg_wall_{}.txt'.format(path), py_seg_wall, fmt='%d', delimiter=' ')
            np.savetxt('metric_seg/gt_seg_wall_{}.txt'.format(path), gt_seg_wall, fmt='%d', delimiter=' ')
            np.savetxt('metric_seg/py_seg_lumen_{}.txt'.format(path), py_seg_lumen, fmt='%d', delimiter=' ')
            np.savetxt('metric_seg/gt_seg_lumen_{}.txt'.format(path), gt_seg_lumen, fmt='%d', delimiter=' ')
            np.savetxt('metric_seg/py_seg_artery_{}.txt'.format(path), py_seg_artery, fmt='%d', delimiter=' ')
            np.savetxt('metric_seg/gt_seg_artery_{}.txt'.format(path), gt_seg_artery, fmt='%d', delimiter=' ')


        print("aveiou:", np.average(self.aveiou))
        print("aveiou_inner:", np.average(self.aveiou_inner))
        print("aveiou_outer:", np.average(self.aveiou_outer))
#dice
        dice = self.dice(py_seg_wall, gt_seg_wall)
        print("dice:", iou)
        dice_lumen = self.dice(py_seg_lumen, gt_seg_lumen)
        print("dice_lumen:", dice_lumen)
        dice_artery = self.dice(py_seg_artery, gt_seg_artery)
        print("dice_artery:", dice_artery)
        if(dice_artery>0):
            self.avedice.append(dice)
            self.avedice_inner.append(dice_lumen)
            self.avedice_outer.append(dice_artery)


        print("avedice:", np.average(self.avedice))
        print("avedice_inner:", np.average(self.avedice_inner))
        print("avedice_outer:", np.average(self.avedice_outer))



        if not iou0check:
            return self.avehausdorff_int,0,0,0

        inp = img_utils.unnormalize_img(batch['inp'][0][15 // 2, :, :], mean[:, :, 15 // 2], std[:, :, 15 // 2])
        output_test = F.softmax(output["prob_map"], dim=1)[-1].cpu().numpy()

        # for BCL
        prob_test_sum = np.sum(output_test,axis = 0)
        mask = np.zeros(output_test.shape)

        mask[1][output_test[1] > snake_config.threshold] = 255
        mask[2][output_test[2] > snake_config.threshold] = 255
        inner = mask[1]
        outer = mask[2]

        fig, ax = plt.subplots(3, 2 + len(ex), figsize=(20, 10))
        fig.tight_layout()
        # ax.axis('off')

        ax[0, 0].imshow(inner, cmap='gray')
        ax[1, 0].imshow(outer, cmap='gray')
        if "prob_map_seg" in output.keys():
            ax[2, 0].imshow(mask_seg_wall,cmap='gray')
        ax[2, 3].imshow(gt_seg_rgb)
        # ax[2, 2].imshow(mask_plq_only,cmap='Reds')
        ax[2, 1].imshow(inp, cmap='gray')
        ax[2, 4].imshow(gt_seg_rgb)
        fig.tight_layout()

        for i in range(len(ex)):
            ax[0, i + 2].imshow(inp, cmap='gray')
            ax[1, i + 2].imshow(inp, cmap='gray')

        ax[1, 1].imshow(inp, cmap='gray')
        ax[0, 1].imshow(inp, cmap='gray')


        colors_0 = np.array([
            [10, 127, 255],
            [255, 127, 14]
        ]) / 255.
        colors_0 = cycle(colors_0)
        if len(ex) == 0:
            print("length has zero")
        for i in range(len(ex)):
            for j in range(len(ex[i])):
                ex1 = ex[i].detach().cpu().numpy() * snake_config.down_ratio
                color = next(colors_0).tolist()

                poly = ex1[j]
                if i == 0:
                    poly = ex1[j] / snake_config.ro

                poly = np.append(poly, [poly[0]], axis=0)
                ax[1, 2 + i].plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)


        colors_0 = np.array([
            [10, 127, 255],
            [255, 127, 14]

        ]) / 255.
        colors_0 = cycle(colors_0)
        if len(ex) == 0:
            print("length has zero")
        for j in range(len(ex[-1])):
            ex1 = ex[-1].detach().cpu().numpy() * snake_config.down_ratio
            color = next(colors_0).tolist()
            poly = ex1[j]

            poly = np.append(poly, [poly[0]], axis=0)
            ax[2, 0].plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)
            ax[2, 2].plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)
            ax[2, 3].plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)


        ax[1, 4].set_title('hdf_0: {:.3f} '.format(float(hdf[0])) + 'hdf_1: {:.3f} '.format(float(hdf[1])))
        ax[2, 3].set_title('p_involve_in: {:.3f} '.format(float(p_involve_in)) + 'p_involve_out: {:.3f} '.format(float(p_involve_out)))

        colors_gt = np.array([
            [31, 119, 180],
            [0, 0, 0]
        ]) / 255.
        for i in range(len(gt)):
            color1 = colors_gt[0].tolist()
            if ct_cls[i]:
                color1 = colors_gt[1].tolist()

            poly1 = gt[i]
            poly1 = np.append(poly1, [poly1[0]], axis=0)
            ax[0, 1].plot(poly1[:, 0], poly1[:, 1], color=color1, linewidth=2)
        colors_gt = np.array([
            [31, 119, 180],
            [0, 0, 0]
        ]) / 255.
        for j in range(len(ex)):
            for i in range(len(gt_py)):
                color1 = colors_gt[0].tolist()
                if ct_cls[i]:
                    color1 = colors_gt[1].tolist()
                poly1 = gt[i]
                poly1 = np.append(poly1, [poly1[0]], axis=0)
                ax[0, 2 + j].plot(poly1[:, 0], poly1[:, 1], color=color1, linewidth=2)
        colors_gt = np.array([
            [70, 150, 130],
            [20, 70, 100]
        ]) / 255.
        for i in range(len(i_it_py)):
            color1 = colors_gt[0].tolist()
            if ct_cls[i]:
                color1 = colors_gt[1].tolist()
            poly1 = i_it_py[i]
            poly1 = np.append(poly1, [poly1[0]], axis=0)
            ax[0, 1].plot(poly1[:, 0], poly1[:, 1], color=color1, linewidth=2)

        colors_gt = np.array([
            [70, 150, 130],
            [20, 70, 100]
        ]) / 255.
        for i in range(len(i_it_py)):
            color1 = colors_gt[0].tolist()
            if ct_cls[i]:
                color1 = colors_gt[1].tolist()
            poly1 = i_it_py[i]
            poly1 = np.append(poly1, [poly1[0]], axis=0)
            ax[0, 1].plot(poly1[:, 0], poly1[:, 1], color=color1, linewidth=2)
        colors_gt = np.array([
            [70, 150, 130],
            [20, 70, 100]
        ]) / 255.

        for i in range(len(evo_i_it_py)):
            color1 = colors_gt[0].tolist()
            if cls[i]:
                color1 = colors_gt[1].tolist()
            poly1 = evo_i_it_py[i]
            poly1 = np.append(poly1, [poly1[0]], axis=0)
            ax[1, 1].plot(poly1[:, 0], poly1[:, 1], color=color1, linewidth=2)
        for i in range(len(evo_i_it_py)):
            color1 = colors_gt[0].tolist()
            if cls[i]:
                color1 = colors_gt[1].tolist()
            poly1 = evo_i_it_py[i]
            poly1 = np.append(poly1, [poly1[0]], axis=0)
            ax[i, 0].plot(poly1[:, 0], poly1[:, 1], color=color1, linewidth=2)

        path = batch["path"][0].split('/')[5] + '+' + batch["path"][0].split('/')[6] + '+' + \
               batch["path"][0].split('/')[9]
        from os.path import exists
        from os import mkdir
        if not exists('ioucheck'):
            mkdir('ioucheck')
        plt.savefig("ioucheck/{}.png".format(path))
        plt.close('all')

        return self.avehausdorff_int,self.class_ct,self.ill_amount,self.sort_hdf

    def boundary(self, output, batch, id):
        # print("batch['inp'][0][15//2,:,:]:", batch['inp'][0][15 // 2, :, :].shape)
        from os.path import exists
        from os import mkdir
        from os.path import join
        from lib.config import cfg, args
        from lib.utils.snake.snake_gcn_utils import uniform_upsample
        from PIL import Image

        boundir = '/data/ugui0/antonio-t/BOUND/patients-ellipse-tiff'
        # boundir = 'config-3_patients-100'
        if not exists(boundir):
            mkdir(boundir)
        boundir = join(boundir, batch["path"][0].split('/')[-6])
        if not exists(boundir):
            mkdir(boundir)
        boundir = join(boundir, batch["path"][0].split('/')[-5])
        if not exists(boundir):
            mkdir(boundir)


        boundir = join(boundir, 'boundary')
        if not exists(boundir):
            mkdir(boundir)
        boundir = join(boundir,  batch["path"][0].split('/')[-4])
        if not exists(boundir):
            mkdir(boundir)
        slice = "%03d" % (int(batch["path"][0].split('/')[-1]) )
        temp = output['py'][-1].unsqueeze(0)
        py = uniform_upsample(temp, 960)
        py = py.squeeze(0)
        py = py.detach().cpu().numpy() * snake_config.down_ratio
        cls = output['cls']
        mask = np.zeros([96, 96])
        for i in range(len(py)):
            print("len(py):",len(py))
            for ord in py[i]:
                if ord[0]>95:
                    ord[0]=95
                if ord[0]<0:
                    ord[0]=0
                if ord[1]>95:
                    ord[1]=95
                if ord[1]<0:
                    ord[1]=0
                if cls[i] == 0:
                    if mask[int(ord[1])][int(ord[0])] == 3:
                        continue
                    if mask[int(ord[1])][int(ord[0])] == 2:
                        mask[int(ord[1])][int(ord[0])] = 3
                        continue
                    mask[int(ord[1])][int(ord[0])] = 1
                if cls[i] == 1:
                    if mask[int(ord[1])][int(ord[0])] == 3:
                        continue
                    if mask[int(ord[1])][int(ord[0])] == 1:
                        mask[int(ord[1])][int(ord[0])] = 3
                        continue
                    mask[int(ord[1])][int(ord[0])] = 2
        im = Image.fromarray( mask)
        im.save(join(boundir, slice) + '.tif')

        return

    def feature_save(self, output, batch, id):
        # print("batch['inp'][0][15//2,:,:]:", batch['inp'][0][15 // 2, :, :].shape)
        from os.path import exists
        from os import mkdir
        from os.path import join
        from lib.config import cfg, args
        from lib.utils.snake.snake_gcn_utils import uniform_upsample

        boundir = '/data/ugui0/antonio-t/BOUND/feature_risk_label_all'
        if not exists(boundir):
            mkdir(boundir)
        boundir = '/data/ugui0/antonio-t/BOUND/feature_risk_label_all/{}'.format(cfg.test.state)
        if not exists(boundir):
            mkdir(boundir)
        boundir = join(boundir, batch["path"][0].split('/')[5])
        if not exists(boundir):
            mkdir(boundir)
        boundir = join(boundir, batch["path"][0].split('/')[6])
        if not exists(boundir):
            mkdir(boundir)
        slice = batch["path"][0].split('/')[9]
        temp = output['py'][-1].cpu().numpy()
        if len(temp) == 2:
            thick, _ = self.myocardial_thickness(temp[0], temp[1])
        if len(temp) == 1:
            thick, _ = self.myocardial_thickness(temp[0], temp[0])

    def fill(self, data, start_coords, fill_value):
        """
        Flood fill algorithm

        Parameters
        ----------
        data : (M, N) ndarray of uint8 type
            Image with flood to be filled. Modified inplace.
        start_coords : tuple
            Length-2 tuple of ints defining (row, col) start coordinates.
        fill_value : int
            Value the flooded area will take after the fill.

        Returns
        -------
        None, ``data`` is modified inplace.
        """
        xsize, ysize = data.shape
        orig_value = data[start_coords[0], start_coords[1]]

        stack = set(((start_coords[0], start_coords[1]),))
        if fill_value == orig_value:
            raise ValueError("Filling region with same value "
                             "already present is unsupported. "
                             "Did you already fill this region?")
        while stack:
            x, y = stack.pop()
            if data[x, y] == orig_value:
                data[x, y] = fill_value
                if x > 0:
                    stack.add((x - 1, y))
                if x < (xsize - 1):
                    stack.add((x + 1, y))
                if y > 0:
                    stack.add((x, y - 1))
                if y < (ysize - 1):
                    stack.add((x, y + 1))

    # feature_risk_label
    def mask_dis_save(self, batch, id):
        from os.path import exists
        from os import mkdir
        from os.path import join
        import scipy
        from lib.config import cfg, args
        from lib.utils.snake.snake_gcn_utils import uniform_upsample

        boundir = '/data/ugui0/antonio-t/BOUND/mask_dis'
        # boundir = '/data/ugui0/antonio-t/BOUND/feature_risk_label_config3'
        # boundir = 'config-3_patients-100'
        if not exists(boundir):
            mkdir(boundir)
        # boundir = '/data/ugui0/antonio-t/BOUND/mask_dis/{}'.format(cfg.test.state)
        # boundir = '/data/ugui0/antonio-t/BOUND/feature_risk_label_config3/{}'.format(cfg.test.state)
        # boundir = 'config-3_patients-100/{}'.format(cfg.test.state)
        if not exists(boundir):
            mkdir(boundir)
        boundir = join(boundir, batch["path"][0].split('/')[5])
        if not exists(boundir):
            mkdir(boundir)
        boundir = join(boundir, batch["path"][0].split('/')[6])
        if not exists(boundir):
            mkdir(boundir)
        slice = batch["path"][0].split('/')[9]
        mask = batch['mask'][0].cpu().numpy()
        # print('mask_shape:',mask.shape)
        # sys.exit()
        # temp = output['py'][-1].cpu().numpy()
        # if len(temp) == 2:
        #     thick, _ = self.myocardial_thickness(temp[0], temp[1])
        # if len(temp) == 1:
        #     thick, _ = self.myocardial_thickness(temp[0], temp[0])

        mask_seg = np.ones(mask.shape)
        mask_1 = np.ones(mask.shape)
        # 1 for inner mask_1 is segmentation
        mask_back = np.ones(mask.shape)
        mask_wall = np.ones(mask.shape)
        mask_2 = np.ones(mask.shape)

        mask_dis = np.zeros([3, 96, 96])

        # 2 for outer mask_2 is segmentation

        mask_1[mask == 1] = 0
        mask_2[mask == 2] = 0
        self.fill(mask_1, (0, 0), 0)
        self.fill(mask_2, (0, 0), 0)
        mask_back[mask_2 == 1] = 0
        mask_wall = mask_2 - mask_1

        mask_seg[mask_back == 1] = 0
        # background
        mask_seg[mask_wall == 1] = 1
        #  artery wall
        mask_seg[mask_1 == 1] = 2

        ex_in = np.argwhere(mask == 1)
        ex_out = np.argwhere(mask == 2)
        for i in range(96):
            for j in range(96):
                distance_in = []
                for k in ex_in:
                    a = np.array(k)
                    b = np.array([i, j])
                    dst = scipy.spatial.distance.euclidean(a, b)
                    distance_in = np.append(distance_in, dst)
                distance_in = np.array(distance_in)
                if len(distance_in):
                    min_dist_in = np.min(distance_in)
                else:
                    min_dist_in = scipy.spatial.distance.euclidean(np.array([47, 47]), np.array([i, j]))
                # print("min_dist_in:", min_dist_in)

                if mask_seg[i][j] == 2:
                    mask_dis[0][i][j] = -1 * min_dist_in
                else:
                    mask_dis[0][i][j] = min_dist_in

                distance_out = []
                for z in ex_out:
                    a = np.array(z)
                    b = np.array([i, j])
                    dst = scipy.spatial.distance.euclidean(a, b)
                    distance_out = np.append(distance_out, dst)
                distance_out = np.array(distance_out)
                if len(distance_out):
                    min_dist_out = np.min(distance_out)
                else:
                    min_dist_out = scipy.spatial.distance.euclidean(np.array([47, 47]), b)

                if mask_seg[i][j] == 0:
                    mask_dis[1][i][j] = min_dist_out
                else:
                    mask_dis[1][i][j] = -1 * min_dist_out

        # print("temp_shape:",temp.shape)
        # sys.exit()
        # thick = batch['thickness_set'].cpu().numpy()
        # risk_label = batch['risk_label'].cpu().numpy()
        # # print('thick:',thick)
        # print('thick_shape:', thick.shape)
        # print('risk_label:', risk_label)
        # print('risk_label_shape:', risk_label.shape)
        # sys.exit()
        # np.savetxt(join(boundir,slice)+'_feature.txt',thick)
        # np.savetxt(join(boundir, slice) + '_label.txt', risk_label)
        np.savetxt(join(boundir, slice) + '_inner.txt', mask_dis[0])
        np.savetxt(join(boundir, slice) + '_outer.txt', mask_dis[1])

        # np.savetxt("boundary_fdata_3_rgb/{}.txt".format(path), mask, fmt="%d")
        # cv2.imwrite(join(boundir,slice)+'.tiff',mask)

        return

    def hausdorff_distance(self, output, gt):
        """ Compute the Hausdorff Distance function between the estimated probability map
        and ground truth points.
        :param output: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        """
        # if len(output.shape) == 5:  # 3D volume
        #     output = output.permute(0, 2, 1, 3, 4)
        #     output = output.contiguous().view(-1, *output.size()[2:])  # combine first 2 dims
        #     gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims
        # check_0= 0
        # check_1=0
        # for i in label:
        #     if i==0:
        #         check_0=1
        #     if i==1:
        #         check_1=1
        # print("check_0:",check_0,"check_1:",check_1)
        # if not check_1 or not check_0:
        #     print(colored("check warning!","red"))
        #
        #     print("check_0:", check_0, "check_1:", check_1)
        #     return
        print("gt:", gt.shape)
        print("output:", output.shape)

        batch_size, height, width = output.shape
        # print("batch_size, height, width:",batch_size," ", height," ", width)
        #    assert batch_size == len(gt), 'output and GT must have the same size'
        # here we consider inner bound and outer bound respectively
        hausdorff_computer = sitk.HausdorffDistanceImageFilter()
        res_bounds_lst = [[] for _ in range(2)]
        for b in range(batch_size):
            output_b, gt_b = output[b], gt[b]
            # print("gt_bb_size:", gt_b.shape)
            # print("output_bb_size:", output_b.shape)
            cv2.imwrite("gt_b_new.png", gt_b * 255)
            cv2.imwrite("output_b_new.png", output_b * 255)
            # print(type(gt_bb));print(gt_bb.shape)
            # print(type(output_bb));print(output_bb.shape)
            # gt_bb = sitk.GetImageFromArray(gt_b.astype(float), isVector=False)
            # output_bb = sitk.GetImageFromArray(output_b.astype(float), isVector=False)
            if not np.any(gt_b) or not np.any(output_b):
                # res_bounds_lst[bound_inx].append(0)
                res_bounds_lst[b].append(0.0)
                continue
            # print("before_hdf")
            hausdorff = hd(output_b, gt_b)
            print("hausdorff:",hausdorff)

            # hausdorff_computer.Execute(gt_bb, output_bb)
            # print("after_hdf")
            # avgHausdorff = hausdorff_computer.GetAverageHausdorffDistance()
            # hausdorff = hausdorff_computer.GetHausdorffDistance()
            # Change between hausdorff and average
            res_bounds_lst[b].append(hausdorff)
            # print("finish")
        # for b in range(batch_size):
        #     output_b, gt_b = output[b], gt[b]
        #     for bound_inx in range(1, n_channel):
        #         gt_bb = (gt_b == bound_inx)
        #         output_bb = output_b[bound_inx - 1]
        #         # print(type(gt_bb));print(gt_bb.shape)
        #         # print(type(output_bb));print(output_bb.shape)
        #         gt_bb = sitk.GetImageFromArray(gt_bb.astype(float), isVector=False)
        #         output_bb = sitk.GetImageFromArray(output_bb.astype(float), isVector=False)
        #         if not np.any(gt_bb) or not np.any(output_bb):
        #             res_bounds_lst[bound_inx].append(0)
        #             continue
        #         hausdorff_computer.Execute(gt_bb, output_bb)
        #         avgHausdorff = hausdorff_computer.GetAverageHausdorffDistance()
        #         hausdorff = hausdorff_computer.GetHausdorffDistance()
        #         # Change between hausdorff and average
        #         res_bounds_lst[bound_inx].append(hausdorff)
        res_bounds = [np.stack(res_bounds_lst[i]) for i in range(2)]
        # print("hdf_ok")
        return res_bounds
        sub_hausdorff_0 = np.average(res_bounds[0])
        sub_hausdorff_1 = np.average(res_bounds[1])

    def myocardial_thickness(self, ords_in, ords_ex):
        """
        Calculate myocardial thickness of mid-slices, excluding a few apex and basal slices
        since myocardium is difficult to identify
        """
        # label_obj = nib.load(data_path)
        # myocardial_mask = (label_obj.get_data() == myo_label)
        # # pixel spacing in X and Y
        # pixel_spacing = label_obj.header.get_zooms()[:2]
        # assert pixel_spacing[0] == pixel_spacing[1]
        #
        # holes_filles = np.zeros(myocardial_mask.shape)
        # interior_circle = np.zeros(myocardial_mask.shape)
        #
        # cinterior_circle_edge = np.zeros(myocardial_mask.shape)
        # cexterior_circle_edge = np.zeros(myocardial_mask.shape)
        #
        overall_avg_thickness = []
        overall_std_thickness = []
        # for i in xrange(slices_to_skip[0], myocardial_mask.shape[2] - slices_to_skip[1]):
        #     holes_filles[:, :, i] = ndimage.morphology.binary_fill_holes(myocardial_mask[:, :, i])
        #     interior_circle[:, :, i] = holes_filles[:, :, i] - myocardial_mask[:, :, i]
        #     cinterior_circle_edge[:, :, i] = feature.canny(interior_circle[:, :, i])
        #     cexterior_circle_edge[:, :, i] = feature.canny(holes_filles[:, :, i])
        #     # patch = 64
        #     # utils.imshow(data_augmentation.resize_image_with_crop_or_pad(myocardial_mask[:,:,i], patch, patch),
        #     #     data_augmentation.resize_image_with_crop_or_pad(holes_filles[:,:,i], patch, patch),
        #     #     data_augmentation.resize_image_with_crop_or_pad(interior_circle[:,:,i], patch,patch ),
        #     #     data_augmentation.resize_image_with_crop_or_pad(cinterior_circle_edge[:,:,i], patch, patch),
        #     #     data_augmentation.resize_image_with_crop_or_pad(cexterior_circle_edge[:,:,i], patch, patch),
        #     #     title= ['Myocardium', 'Binary Hole Filling', 'Left Ventricle Cavity', 'Interior Contour', 'Exterior Contour'], axis_off=True)
        # x_in, y_in = np.where(cinterior_circle_edge[:, :, i] != 0)
        # number_of_interior_points = len(x_in)
        #     # print (len(x_in))
        # x_ex, y_ex = np.where(cexterior_circle_edge[:, :, i] != 0)
        # number_of_exterior_points = len(x_ex)
        # print (len(x_ex))
        # if len(x_ex) and len(x_in) != 0:
        total_distance_in_slice = []
        for z in range(len(ords_in)):
            distance = []
            for k in range(len(ords_ex)):
                a = ords_in[z]
                a = np.array(a)
                # print a
                b = ords_ex[k]
                b = np.array(b)
                # dst = np.linalg.norm(a-b)
                dst = scipy.spatial.distance.euclidean(a, b)
                # pdb.set_trace()
                # if dst == 0:
                #     pdb.set_trace()
                distance = np.append(distance, dst)
            distance = np.array(distance)
            min_dist = np.min(distance)
            total_distance_in_slice = np.append(total_distance_in_slice, min_dist)
            total_distance_in_slice = np.array(total_distance_in_slice)

        average_distance_in_slice = np.mean(total_distance_in_slice)
        # overall_avg_thickness = np.append(overall_avg_thickness, average_distance_in_slice)
        #
        # std_distance_in_slice = np.std(total_distance_in_slice)
        # overall_std_thickness = np.append(overall_std_thickness, std_distance_in_slice)

        # print (overall_avg_thickness)
        # print (overall_std_thickness)
        # print (pixel_spacing[0])
        return total_distance_in_slice, average_distance_in_slice

    def visualize(self, output, batch, id):
        # self.visualize_ex(output, batch)
        hdf=self.visualize_training_box(output, batch, id)
        return hdf
    # def boundary(self, output, batch,id):
    #     # self.visualize_ex(output, batch)
    #     self.visualize_training_box(output, batch,id)
