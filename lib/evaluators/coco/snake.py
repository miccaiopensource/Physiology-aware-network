import os
import cv2
import json
import numpy as np
from lib.utils.snake import snake_config, snake_cityscapes_utils, snake_eval_utils, snake_poly_utils
from external.cityscapesscripts.evaluation import evalInstanceLevelSemanticLabeling
import pycocotools.mask as mask_util
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from lib.config import cfg
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.utils import data_utils
from medpy.metric.binary import hd95, asd
import SimpleITK as sitk
from termcolor import colored

class Evaluator:
    def __init__(self, result_dir):
        self.results = []
        self.img_ids = []
        self.cat_ids = []
        self.aps = []
        self.hausdorff=[[] for _ in range(2)]
        self.avehausdorff = []

        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.data_root = args['data_root']
        self.coco = coco.COCO(self.ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
    def hausdorff_distance(self,output, gt,label):
        """ Compute the Hausdorff Distance function between the estimated probability map
        and ground truth points.
        :param output: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        """
        # if len(output.shape) == 5:  # 3D volume
        #     output = output.permute(0, 2, 1, 3, 4)
        #     output = output.contiguous().view(-1, *output.size()[2:])  # combine first 2 dims
        #     gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims
        check_0= 0
        check_1=0
        for i in label:
            if i==0:
                check_0=1
            if i==1:
                check_1=1
        print("check_0:",check_0,"check_1:",check_1)
        if not check_1 or not check_0:
            print(colored("check warning!","red"))

            print("check_0:", check_0, "check_1:", check_1)
            return


        batch_size, height, width = output.shape
        #    assert batch_size == len(gt), 'output and GT must have the same size'
        # here we consider inner bound and outer bound respectively
        hausdorff_computer = sitk.HausdorffDistanceImageFilter()
        res_bounds_lst = [[] for _ in range(2)]
        for b in range(batch_size):
            output_b, gt_b = output[b], gt[b]
                # print(type(gt_bb));print(gt_bb.shape)
                # print(type(output_bb));print(output_bb.shape)
            gt_bb = sitk.GetImageFromArray(gt_b.astype(float), isVector=False)
            output_bb = sitk.GetImageFromArray(output_b.astype(float), isVector=False)
            if not np.any(gt_bb) or not np.any(output_bb):
                # res_bounds_lst[bound_inx].append(0)
                continue
            hausdorff_computer.Execute(gt_bb, output_bb)
            avgHausdorff = hausdorff_computer.GetAverageHausdorffDistance()
            hausdorff = hausdorff_computer.GetHausdorffDistance()
                # Change between hausdorff and average
            res_bounds_lst[label[b]].append(hausdorff)
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
        sub_hausdorff_0 = np.average(res_bounds[0])
        sub_hausdorff_1 = np.average(res_bounds[1])
        self.hausdorff[0].append(sub_hausdorff_0)
        self.hausdorff[1].append(sub_hausdorff_1)

        # with open('hausdorff.txt', 'w') as outfile:
        # # I'm writing a header here just for the sake of readability
        #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #
        #     # Iterating througaskh a ndimensional array produces slices along
        #     # the last axis. This is equivalent to data[i,:,:] in this case
        #     np.savetxt(outfile, self.hausdorff, fmt='%-7.2f')

                # Writing out a break to indicate different slices...
        # with open('avehausdorff.txt', 'w') as outfile:
        #     # I'm writing a header here just for the sake of readability
        #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #
        #     # Iterating througaskh a ndimensional array produces slices along
        #     # the last axis. This is equivalent to data[i,:,:] in this case
        #
        #     np.savetxt(outfile, self.avehausdorff, fmt='%-7.2f')
        #
        #     # Writing out a break to indicate different slices...
        #     outfile.write('# New slice\n')


    def evaluate(self, output, batch):
        detection = output['detection']
        # with open("./detection.json", 'w', encoding='utf-8') as json_file:
        #     json.dump(detection, json_file, ensure_ascii=False)

        score = detection[:, 4].detach().cpu().numpy()
        label = detection[:, 5].detach().cpu().numpy().astype(int)
        # with open('labl_score.txt', 'a') as outfile:
        # # I'm writing a header here just for the sake of readability
        #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #
        #
        #     # Iterating througaskh a ndimensional array produces slices along
        #     # the last axis. This is equivalent to data[i,:,:] in this case
        #     np.savetxt(outfile, label, fmt='%-7.2f')
        #
        #         # Writing out a break to indicate different slices...
        #     outfile.write('# New slice\n')# with open('select_py.txt', 'w')
        #         # as outfile:
        #     np.savetxt(outfile, score, fmt='%-7.2f')
        #
        # # Writing out a break to indicate different slices...
        #     outfile.write('# New slice\n')  # with open('select_py.txt', 'w')
        # # as outfile:np.savetxt(outfile, label, fmt='%-7.2f')
        # #
        # #                 # Writing out a break to indicate different slices...
        # #             outfile.write('# New slice\n')# with open('select_py.txt', 'w')
        # #                 # as outfile:
        #

        py_f=[]
        # py_2= output['py'][-2].detach().cpu().numpy()
        # print("py_2:",py_2)
        # with open('outputpy_2.txt', 'w') as outfile:
        # # I'm writing a header here just for the sake of readability
        #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #
        #
        #     # Iterating througaskh a ndimensional array produces slices along
        #     # the last axis. This is equivalent to data[i,:,:] in this case
        #     for data_slice in py_2:
        #         # The formatting string indicates that I'm writing out
        #         # the values in left-justified columns 7 characters in width
        #         # with 2 decimal places.
        #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #
        #         # Writing out a break to indicate different slices...
        #         outfile.write('# New slice\n')# with open('select_py.txt', 'w') as outfile:
        #         # # I'm writing a header here just for the sake of readability
        #         #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #         #
        #         #
        #         #     # Iterating througaskh a ndimensional array produces slices along
        #         #     # the last axis. This is equivalent to data[i,:,:] in this case
        #         #     for data_slice in py_f:
        #         #         # The formatting string indicates that I'm writing out
        #         #         # the values in left-justified columns 7 characters in width
        #         #         # with 2 decimal places.
        #         #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #         #
        #         #         # Writing out a break to indicate different slices...
        #         #         outfile.write('# New slice\n')
        py = output['py'][-1].detach().cpu().numpy()*snake_config.down_ratio
        # with open('outputpy.txt', 'a') as outfile:
        # # I'm writing a header here just for the sake of readability
        #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #
        #     outfile.write('# New OP\n')
        #     # Iterating througaskh a ndimensional array produces slices along
        #     # the last axis. This is equivalent to data[i,:,:] in this case
        #     for data_slice in py:
        #         # The formatting string indicates that I'm writing out
        #         # the values in left-justified columns 7 characters in width
        #         # with 2 decimal places.
        #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #
        #         # Writing out a break to indicate different slices...
        #         outfile.write('# New slice\n')# with open('select_py.txt', 'w') as outfile:
        #
        #         # # I'm writing a header here just for the sake of readability
        #         #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #         #
        #         #
        #         #     # Iterating througaskh a ndimensional array produces slices along
        #         #     # the last axis. This is equivalent to data[i,:,:] in this case
        #         #     for data_slice in py_f:
        #         #         # The formatting string indicates that I'm writing out
        #         #         # the values in left-justified columns 7 characters in width
        #         #         # with 2 decimal places.
        #         #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #         #
        #         #         # Writing out a break to indicate different slices...
        #         #         outfile.write('# New slice\n')
        # print("")
        # for i in range(2):
        #     max = 0
        #     t=0
        #     for j in range(len(detection)):
        #         if detection[j][5]==i:
        #             if detection[j][4]>max:
        #                 max=detection[j][4]
        #                 _j=j
        #                 t=1
        #     if(t):
        #         py_f.append(py[_j])





        # print(py_f)
        # with open('select_py.txt', 'w') as outfile:
        # # I'm writing a header here just for the sake of readability
        #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #
        #
        #     # Iterating througaskh a ndimensional array produces slices along
        #     # the last axis. This is equivalent to data[i,:,:] in this case
        #     for data_slice in py_f:
        #         # The formatting string indicates that I'm writing out
        #         # the values in left-justified columns 7 characters in width
        #         # with 2 decimal places.
        #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #
        #         # Writing out a break to indicate different slices...
        #         outfile.write('# New slice\n')

        #
        # print("gt_poly:",output["i_gt_py"])
        # print("label:", label)
        # print("py:",py)
        # for key, value in batch.items():
        #    print(key)
        # including_key:
        # ct_hm
        # wh
        # ct
        # detection
        # it_ex
        # ex
        # it_py
        # py

        # inp
        # poly
        # meta
        # ct_hm
        # wh
        # ct_cls
        # ct_ind
        # ct_01
        # i_it_4py
        # c_it_4py
        # i_gt_4py
        # c_gt_4py
        # i_it_py
        # c_it_py
        # i_gt_py
        # c_gt_py

        ct_cls = batch['ct_cls'][-1].detach().cpu().numpy()
        # with open('cls_py.txt', 'a') as outfile:
        # # I'm writing a header here just for the sake of readability
        #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #
        #     outfile.write('# New CLS\n')
        #     # Iterating througaskh a ndimensional array produces slices along
        #     # the last axis. This is equivalent to data[i,:,:] in this case
        #     np.savetxt(outfile, ct_cls, fmt='%-7.2f')
        #     outfile.write('# New slice\n')
        #         # # I'm writing a header here just for the sake of readability
        #         #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #         #
        #         #
        #         #     # Iterating througaskh a ndimensional array produces slices along
        #         #     # the last axis. This is equivalent to data[i,:,:] in this case
        #         #     for data_slice in py_f:
        #         #         # The formatting string indicates that I'm writing out
        #         #         # the values in left-justified columns 7 characters in width
        #         #         # with 2 decimal places.
        #         #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #         #
        #         #         # Writing out a break to indicate different slices...
        #         #         outfile.write('# New slice\n')
        # # print('gt_py:',gt_py)
        # with open('gt_py.txt', 'a') as outfile:
        # # I'm writing a header here just for the sake of readability
        #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #
        #     outfile.write('# New GT\n')
        #     # Iterating througaskh a ndimensional array produces slices along
        #     # the last axis. This is equivalent to data[i,:,:] in this case
        #     for data_slice in gt_py:
        #         # The formatting string indicates that I'm writing out
        #         # the values in left-justified columns 7 characters in width
        #         # with 2 decimal places.
        #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #
        #         # Writing out a break to indicate different slices...
        #         outfile.write('# New slice\n')# with open('select_py.txt', 'w') as outfile:
        #         # # I'm writing a header here just for the sake of readability
        #         #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #         #
        #         #
        #         #     # Iterating througaskh a ndimensional array produces slices along
        #         #     # the last axis. This is equivalent to data[i,:,:] in this case
        #         #     for data_slice in py_f:
        #         #         # The formatting string indicates that I'm writing out
        #         #         # the values in left-justified columns 7 characters in width
        #         #         # with 2 decimal places.
        #         #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #         #
        #         #         # Writing out a break to indicate different slices...
        #         #         outfile.write('# New slice\n')
        gt_py = batch['i_gt_py'][-1].detach().cpu().numpy()*snake_config.down_ratio
        # print('gt_py:',gt_py)
        # with open('gt_py_cls.txt', 'a') as outfile:
        #     # I'm writing a header here just for the sake of readability
        #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #
        #     outfile.write('# New GT\n')
        #     # Iterating througaskh a ndimensional array produces slices along
        #     # the last axis. This is equivalent to data[i,:,:] in this case
        #     for data_slice in gt_py:
        #         # The formatting string indicates that I'm writing out
        #         # the values in left-justified columns 7 characters in width
        #         # with 2 decimal places.
        #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #
        #         # Writing out a break to indicate different slices...
        #         outfile.write('# New slice\n')  # with open('select_py.txt', 'w') as outfile:
        #         # # I'm writing a header here just for the sake of readability
        #         #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #         #
        #         #
        #         #     # Iterating througaskh a ndimensional array produces slices along
        #         #     # the last axis. This is equivalent to data[i,:,:] in this case
        #         #     for data_slice in py_f:
        #         #         # The formatting string indicates that I'm writing out
        #         #         # the values in left-justified columns 7 characters in width
        #         #         # with 2 decimal places.
        #         #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #         #
        #         #         # Writing out a break to indicate different slices...
        #         #         outfile.write('# New slice\n')


        if len(py) == 0:
            return

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']
        # print("ori_h,ori_w=",ori_h, ' ', ori_w)
        py = [data_utils.affine_transform(py_, trans_output_inv) for py_ in py]
        gt_py =[data_utils.affine_transform(py_, trans_output_inv) for py_ in gt_py]
        # print("len(py):", len(py),"ori_w:", ori_w,"ori_h:", ori_h)
        py_mask = np.zeros([len(py), ori_w, ori_h])
        gt_py_mask_f = np.zeros([len(py), ori_w, ori_h])
        # print("py_mask_shape:",py_mask.shape)
        gt_py_mask = np.zeros([len(gt_py), ori_w, ori_h])
        for i in range(len(py)):
            for ord in py[i]:
                # print("ord_0:",int(ord[0]),"ord_1:",int(ord[1]),"i:",i)
                py_mask[i][int(ord[0])][int(ord[1])] = 1
            # cv2.imwrite('mask_py.png', py_mask[i] * 255)
        for i in range(len(gt_py)):
            for ord in gt_py[i]:
                # print("ord_0:",int(ord[0]),"ord_1:",int(ord[1]),"i:",i)
                gt_py_mask[i][int(ord[0])][int(ord[1])] = 1
        for i in range(len(py)):
            # print("label[i]:",label[i])
            gt_py_mask_f[i]=gt_py_mask[label[i]]
            # cv2.imwrite('mask_gt_py_f.png', gt_py_mask_f[i] * 255)
        # print("py_mask_shape:",py_mask.shape,"gt_py_mask_shape:",gt_py_mask.shape,"gt_py_mask_f_shape:",gt_py_mask_f.shape,)
        # return py_mask,gt_py_mask_f,label

        self.hausdorff_distance(py_mask,gt_py_mask_f,label)
        ave_0=np.average(self.hausdorff[0])
        ave_1=np.average(self.hausdorff[1])
        print("ave_0:",ave_0,"ave_1:",ave_1)


            # cv2.imwrite('mask_gt_py.png', gt_py_mask[i] * 255)
        # with open('gtf_poly.txt', 'a') as outfile:
        # # I'm writing a header here just for the sake of readability
        #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #
        #     outfile.write('# New GTF\n')
        #     # Iterating througaskh a ndimensional array produces slices along
        #     # the last axis. This is equivalent to data[i,:,:] in this case
        #     for data_slice in gt_py:
        #         # The formatting string indicates that I'm writing out
        #         # the values in left-justified columns 7 characters in width
        #         # with 2 decimal places.
        #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #
        #         # Writing out a break to indicate different slices...
        #         outfile.write('# New slice\n')# with open('gtf_poly.txt', 'w') as outfile:
        #         # # I'm writing a header here just for the sake of readability
        #         #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #         #
        #         #
        #         #     # Iterating througaskh a ndimensional array produces slices along
        #         #     # the last axis. This is equivalent to data[i,:,:] in this case
        #         #     for data_slice in gt_py:
        #         #         # The formatting string indicates that I'm writing out
        #         #         # the values in left-justified columns 7 characters in width
        #         #         # with 2 decimal places.
        #         #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #         #
        #         #         # Writing out a break to indicate different slices...
        #         #         outfile.write('# New slice\n')
        # with open('py_mask.txt', 'a') as outfile:
        # # I'm writing a header here just for the sake of readability
        #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #
        #
        #     # Iterating througaskh a ndimensional array produces slices along
        #     # the last axis. This is equivalent to data[i,:,:] in this case
        #     outfile.write('# New PY_MASK\n')
        #     for data_slice in py_mask:
        #         # The formatting string indicates that I'm writing out
        #         # the values in left-justified columns 7 characters in width
        #         # with 2 decimal places.
        #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #
        #         # Writing out a break to indicate different slices...
        #         outfile.write('# New slice\n')# with open('gtf_poly.txt', 'w') as outfile:
                # # I'm writing a header here just for the sake of readability
                #     # Any line starting with "#" will be ignored by numpy.loadtxt
                #
                #
                #     # Iterating througaskh a ndimensional array produces slices along
                #     # the last axis. This is equivalent to data[i,:,:] in this case
                #     for data_slice in gt_py:
                #         # The formatting string indicates that I'm writing out
                #         # the values in left-justified columns 7 characters in width
                #         # with 2 decimal places.
                #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
                #
                #         # Writing out a break to indicate different slices...
                #         outfile.write('# New slice\n')        with open('gtf_poly.txt', 'a') as outfile:
                #         # I'm writing a header here just for the sake of readability
                #             # Any line starting with "#" will be ignored by numpy.loadtxt
                #
                #
                #             # Iterating througaskh a ndimensional array produces slices along
                #             # the last axis. This is equivalent to data[i,:,:] in this case
                #             for data_slice in gt_py:
                #                 # The formatting string indicates that I'm writing out
                #                 # the values in left-justified columns 7 characters in width
                #                 # with 2 decimal places.
                #                 np.savetxt(outfile, data_slice, fmt='%-7.2f')
                #
                #                 # Writing out a break to indicate different slices...
                #                 outfile.write('# New slice\n')# with open('gtf_poly.txt', 'w') as outfile:
                #                 # # I'm writing a header here just for the sake of readability
                #                 #     # Any line starting with "#" will be ignored by numpy.loadtxt
                #                 #
                #                 #
                #                 #     # Iterating througaskh a ndimensional array produces slices along
                #                 #     # the last axis. This is equivalent to data[i,:,:] in this case
                #                 #     for data_slice in gt_py:
                #                 #         # The formatting string indicates that I'm writing out
                #                 #         # the values in left-justified columns 7 characters in width
                #                 #         # with 2 decimal places.
                #                 #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
                #                 #
                #                 #         # Writing out a break to indicate different slices...
                #                 #         outfile.write('# New slice\n')
        # print("final_poly:",py)
        # print("gt_py",gt_py)
        # with open('gtf_poly.txt', 'w') as outfile:
        # # I'm writing a header here just for the sake of readability
        #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #
        #
        #     # Iterating througaskh a ndimensional array produces slices along
        #     # the last axis. This is equivalent to data[i,:,:] in this case
        #     for data_slice in gt_py:
        #         # The formatting string indicates that I'm writing out
        #         # the values in left-justified columns 7 characters in width
        #         # with 2 decimal places.
        #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #
        #         # Writing out a break to indicate different slices...
        #         outfile.write('# New slice\n')

        # with open('f_poly.txt', 'w') as outfile:
        # # I'm writing a header here just for the sake of readability
        #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #
        #
        #     # Iterating througaskh a ndimensional array produces slices along
        #     # the last axis. This is equivalent to data[i,:,:] in this case
        #     for data_slice in py:
        #         # The formatting string indicates that I'm writing out
        #         # the values in left-justified columns 7 characters in width
        #         # with 2 decimal places.
        #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
        #
        #         # Writing out a break to indicate different slices...
        #         outfile.write('# New slice\n')
        #
        # rles = snake_eval_utils.coco_poly_to_rle(py, ori_h, ori_w)
        #
        # coco_dets = []
        # for i in range(len(rles)):
        #     detection = {
        #         'image_id': img_id,
        #         'category_id': self.contiguous_category_id_to_json_id[label[i]],
        #         'segmentation': rles[i],
        #         'score': float('{:.2f}'.format(score[i]))
        #     }
        #     coco_dets.append(detection)  # 数据
        # with open("./det.json", 'w', encoding='utf-8') as json_file:
        #     json.dump(coco_dets, json_file, ensure_ascii=False)
        #
        #
        # self.results.extend(coco_dets)
        # self.img_ids.append(img_id)
    def evaluate_hdf(self, output, batch):
        detection = output['detection']
        # with open("./detection.json", 'w', encoding='utf-8') as json_file:
        #     json.dump(detection, json_file, ensure_ascii=False)

        score = detection[:, 4].detach().cpu().numpy()
        label = detection[:, 5].detach().cpu().numpy().astype(int)

        py_f=[]
        py = output['py'][-1].detach().cpu().numpy()







        # for key, value in batch.items():
        #    print(key)
        # including_key:
        # ct_hm
        # wh
        # ct
        # detection
        # it_ex
        # ex
        # it_py
        # py

        # inp
        # poly
        # meta
        # ct_hm
        # wh
        # ct_cls
        # ct_ind
        # ct_01
        # i_it_4py
        # c_it_4py
        # i_gt_4py
        # c_gt_4py
        # i_it_py
        # c_it_py
        # i_gt_py
        # c_gt_py

        gt_py=batch['i_gt_py'][-1].detach().cpu().numpy()
        print('gt_py:',gt_py)


        if len(py) == 0:
            return

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']
        py = [data_utils.affine_transform(py_, trans_output_inv) for py_ in py]
        gt_py =[data_utils.affine_transform(py_, trans_output_inv) for py_ in gt_py]
        py_mask=np.zeros_like(len(py),ori_w,ori_h)
        gt_py_mask=np.zeros_like(len(gt_py),ori_w,ori_h)
        for i in py:
            for ord in i:
                py_mask[i][int(ord[0])][int(ord[1])]=1

        rles = snake_eval_utils.coco_poly_to_rle(py, ori_h, ori_w)

        coco_dets = []
        for i in range(len(rles)):
            detection = {
                'image_id': img_id,
                'category_id': self.contiguous_category_id_to_json_id[label[i]],
                'segmentation': rles[i],
                'score': float('{:.2f}'.format(score[i]))
            }
            coco_dets.append(detection)  # 数据
        with open("./det.json", 'w', encoding='utf-8') as json_file:
            json.dump(coco_dets, json_file, ensure_ascii=False)



        self.results.extend(coco_dets)
        self.img_ids.append(img_id)



    def hausdorff_distance_orig(output, gt):
        """ Compute the Hausdorff Distance function between the estimated probability map
        and ground truth points.
        :param output: (B x C x H x W) Tensor of estimated probability map with multiple channels
        :param gt: (B x H x W) Tensor of the GT annotation
        """
        if len(output.shape) == 5:  # 3D volume
            output = output.permute(0, 2, 1, 3, 4)
            output = output.contiguous().view(-1, *output.size()[2:])  # combine first 2 dims
            gt = gt.contiguous().view(-1, *gt.size()[2:])  # combine first 2 dims
        batch_size, n_channel, height, width = output.shape
        n_channel = n_channel + 1
        #    assert batch_size == len(gt), 'output and GT must have the same size'
        # here we consider inner bound and outer bound respectively
        hausdorff_computer = sitk.HausdorffDistanceImageFilter()
        res_bounds_lst = [[] for _ in range(0, n_channel)]
        for b in range(batch_size):
            output_b, gt_b = output[b], gt[b]
            for bound_inx in range(1, n_channel):
                gt_bb = (gt_b == bound_inx)
                output_bb = output_b[bound_inx - 1]
                # print(type(gt_bb));print(gt_bb.shape)
                # print(type(output_bb));print(output_bb.shape)
                gt_bb = sitk.GetImageFromArray(gt_bb.astype(float), isVector=False)
                output_bb = sitk.GetImageFromArray(output_bb.astype(float), isVector=False)
                if not np.any(gt_bb) or not np.any(output_bb):
                    res_bounds_lst[bound_inx].append(0)
                    continue
                hausdorff_computer.Execute(gt_bb, output_bb)
                avgHausdorff = hausdorff_computer.GetAverageHausdorffDistance()
                hausdorff = hausdorff_computer.GetHausdorffDistance()
                # Change between hausdorff and average
                res_bounds_lst[bound_inx].append(hausdorff)
        res_bounds = [np.stack(res_bounds_lst[i]) for i in range(1, n_channel)]
        return res_bounds
    def summarize(self):
        with open('avehausdorff.txt', 'w') as outfile:
    # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt


        # Iterating througaskh a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case

            np.savetxt(outfile, self.avehausdorff, fmt='%-7.2f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
        # json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        # coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        # coco_eval = COCOeval(self.coco, coco_dets, 'segm')
        # coco_eval.params.imgIds = self.img_ids
        # coco_eval.evaluate()
        # coco_eval.accumulate()
        # coco_eval.summarize()
        # self.results = []
        # self.img_ids = []
        # self.aps.append(coco_eval.stats[0])
        # return {'ap': coco_eval.stats[0]}

    def volumewise_hd95(self,preds, targets, return_slicewise_hdf=False, n_classes=3):
        """ calculate volume-wise 95 percentile Hausdorff distance
        :param preds: Array/Tensor with size [B, D, H, W] or [B, H, W]
        :param targets: Array/Tensor with size [B, D, H, W] or [B, H, W]
        :param return_slicewise_hdf: bool, whether return slice-wise hdf or not
        :para n_classes, int, how many classes for calculating HDF
        """

        batch_res = []
        if not isinstance(preds, np.ndarray):  # set1 and set2 are tensors
            preds = preds.data.cpu().numpy()
            targets = targets.data.cpu().numpy()

        assert len(preds) == len(targets), \
            "length of preds should be equal to that of targets, but got {} and {}".format(len(preds), len(targets))

        if preds.ndim == 4 and targets.ndim == 4:  # convert 3D to 2D if preds and targets are volumes
            preds = np.reshape(preds, (preds.shape[0] * preds.shape[1], *preds.shape[2:]))
            targets = np.reshape(targets, (targets.shape[0] * targets.shape[1], *targets.shape[2:]))

        for pred, target in zip(preds, targets):
            if np.sum(target) != 0:
                slice_hd95 = slicewise_hd95(pred, target, n_classes)
                batch_res.append(slice_hd95)

        mean_hd95 = sum(batch_res) / len(batch_res)

        if return_slicewise_hdf:
            return mean_hd95, batch_res
        else:
            return mean_hd95

    def slicewise_hd95(slef,pred, target, n_classes=3):
        """ calculate Average Hausdorff distance between pred and target of single image
        :param pred: ndarray with size [H, W],  predicted bound
        :param target: ndarray with size [H, W], ground truth
        :param n_classes: int, # of classes
        """
        max_hd95 = 2 * target.shape[0]
        slice_res = []
        for c_inx in range(1, n_classes):
            pred_cinx = (pred == c_inx)
            target_cinx = (target == c_inx)
            if np.sum(target_cinx) != 0:
                if np.sum(pred_cinx) == 0:
                    res = max_hd95
                else:
                    res = hd95(pred_cinx, target_cinx)

                slice_res.append(res)

        mean_res = sum(slice_res) / len(slice_res)

        return mean_res


class DetectionEvaluator:
    def __init__(self, result_dir):
        self.results = []
        self.img_ids = []
        self.aps = []

        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))

        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.data_root = args['data_root']
        self.coco = coco.COCO(self.ann_file)

        self.json_category_id_to_contiguous_id = {
            v: i for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def evaluate(self, output, batch):
        detection = output['detection']
        detection = detection[0] if detection.dim() == 3 else detection
        box = detection[:, :4].detach().cpu().numpy() * snake_config.down_ratio
        score = detection[:, 4].detach().cpu().numpy()
        label = detection[:, 5].detach().cpu().numpy().astype(int)

        img_id = int(batch['meta']['img_id'][0])
        center = batch['meta']['center'][0].detach().cpu().numpy()
        scale = batch['meta']['scale'][0].detach().cpu().numpy()

        if len(box) == 0:
            return

        h, w = batch['inp'].size(2), batch['inp'].size(3)
        trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
        img = self.coco.loadImgs(img_id)[0]
        ori_h, ori_w = img['height'], img['width']

        coco_dets = []
        for i in range(len(label)):
            box_ = data_utils.affine_transform(box[i].reshape(-1, 2), trans_output_inv).ravel()
            box_[2] -= box_[0]
            box_[3] -= box_[1]
            box_ = list(map(lambda x: float('{:.2f}'.format(x)), box_))
            detection = {
                'image_id': img_id,
                'category_id': self.contiguous_category_id_to_json_id[label[i]],
                'bbox': box_,
                'score': float('{:.2f}'.format(score[i]))
            }
            coco_dets.append(detection)

        self.results.extend(coco_dets)
        self.img_ids.append(img_id)

    def summarize(self):
        json.dump(self.results, open(os.path.join(self.result_dir, 'results.json'), 'w'))
        coco_dets = self.coco.loadRes(os.path.join(self.result_dir, 'results.json'))
        coco_eval = COCOeval(self.coco, coco_dets, 'bbox')
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.results = []
        self.img_ids = []
        self.aps.append(coco_eval.stats[0])
        return {'ap': coco_eval.stats[0]}


Evaluator = Evaluator if cfg.segm_or_bbox == 'segm' else DetectionEvaluator
