import torch.utils.data as data
from lib.utils.snake import snake_kins_utils, snake_config, visualize_utils
import cv2
import numpy as np
import math
from lib.utils import data_utils
from pycocotools.coco import COCO
import os
import os.path as osp
from skimage import io, transform
from lib.config import cfg, args
from lib.datasets.transforms_medical import make_transforms
import matplotlib.pyplot as plt
from itertools import cycle
import pickle
import sys
import pickle
import json
import torch
import time
from skimage import measure
import scipy
class Dataset(data.Dataset):
    def __init__(self, ann_file,ann_file_nc, data_root,split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split
        self.multi_view=cfg.multi_view

        self.coco = COCO(ann_file)
        self.coco_nc= COCO(ann_file_nc)

        self.anns = np.array(self.coco.getImgIds())
        # self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        # print("ann_file_nc:",ann_file_nc)
        self.coco_nc = COCO(ann_file_nc)
        self.anns_nc = np.array(self.coco_nc.getImgIds())
        # self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id_nc = {v: i for i, v in enumerate(self.coco_nc.getCatIds())}

    def process_info(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        # print("path_info:",path)



        return anno, path, img_id
    def binary_mask_to_polygon(self,binary_mask, tolerance=0):
        polygons = []
        # pad mask to close contours of shapes which start and end at an edge
        # print("binary_mask.shape:",binary_mask.shape)
        padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_binary_mask, 0.5)
        # contours = measure.find_contours(binary_mask, 0.5)
        # print(contours)
        # with open('contour.txt', 'w') as outfile:
        #     # I'm writing a header here just for the sake of readability
        #     # Any line starting with "#" will be ignored by numpy.loadtxt
        #
        #
        #     # Iterating througaskh a ndimensional array produces slices along
        #     # the last axis. This is equivalent to data[i,:,:] in this case
        #     for data_slice in contours:
        #         # The formatting string indicates that I'm writing out
        #         # the values in left-justified columns 7 characters in width
        #         # with 2 decimal places.
        #         np.savetxt(outfile, data_slice, fmt='%s')
        #
        #         # Writing out a break to indicate different slices...
        #         outfile.write('# New slice\n')
        # print("finish countours")

        # contours = np.subtract(contours, 1)
        contours = np.subtract(np.array(contours, dtype=object), 1)
        for contour in contours:
            contour = self.close_contour(contour)
            contour = measure.approximate_polygon(contour, tolerance)
            if len(contour) < 3:
                continue
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            # after padding and subtracting 1 we may get -0.5 points in our segmentation
            segmentation = [0 if i < 0 else i for i in segmentation]
            polygons.append(segmentation)
        return polygons

    def close_contour(self, contour):
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour
    def myocardial_thickness(self,ords_in,ords_ex):
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
    def read_original_data(self, anno, path):
        from os import listdir
        start = time.time()

        if self.multi_view:
            axis_names = ['applicate', 'abscissa', 'ordinate']
        else:
            axis_names = ['applicate']

        # print("path:",path)
        for a_inx, axis_name in enumerate(axis_names):
            image_path_axis = path.replace('ordinate', axis_name)
            mask_path_axis = image_path_axis.replace('image','mask')
            slice_idx=image_path_axis.split('/')[-1]
            image_path_axis = image_path_axis + ".tiff"
            mask_path_axis = mask_path_axis + ".tiff"
            risk_label_path=''
            image_path = ''
            mask_dis_dir = '/data/ugui0/antonio-t/BOUND/mask_dis'
            mask3d_dis_dir = '/data/ugui0/antonio-t/BOUND/mask_dis_tiff_3D'


            mask_dis_dir = os.path.join(mask_dis_dir, path.split('/')[5])
            mask_dis_dir = os.path.join(mask_dis_dir, path.split('/')[6])
            mask3d_dis_dir = os.path.join(mask3d_dis_dir, path.split('/')[5])
            mask3d_dis_dir = os.path.join(mask3d_dis_dir, path.split('/')[6])
            slice = path.split('/')[9]
            slice_3d = "%03d" % (int(path.split('/')[9]) + cfg.interval // 2)
            mask3d_dis_noncal = os.path.join(mask3d_dis_dir , slice_3d)+'_noncal.txt'
            mask3d_path = path.split('/')[5]+'_'+path.split('/')[6]+'_'+ slice_3d
            mask3d_dis_noncal_exist = os.path.join(mask3d_dis_dir, slice_3d) + '_noncal_exist.txt'
            mask3d_dis_noncal = np.loadtxt(mask3d_dis_noncal)
            mask3d_dis_noncal_exist = np.loadtxt(mask3d_dis_noncal_exist)
            # dir = 'exist'
            if mask3d_dis_noncal_exist:
                mask3d_dis_noncal[mask3d_dis_noncal>68]=68
                # np.savetxt(os.path.join(dir,mask3d_path+'exist_noncal.txt'), mask3d_dis_noncal,fmt='%d')
            else:
                mask3d_dis_noncal = mask3d_dis_noncal + 68
                # np.savetxt(os.path.join(dir,mask3d_path+'nonexist_noncal.txt'), mask3d_dis_noncal,fmt='%d')

            mask3d_dis_cal = os.path.join(mask3d_dis_dir, slice_3d) + '_cal.txt'
            mask3d_dis_cal_exist = os.path.join(mask3d_dis_dir, slice_3d) + '_cal_exist.txt'
            mask3d_dis_cal = np.loadtxt(mask3d_dis_cal)
            mask3d_dis_cal_exist = np.loadtxt(mask3d_dis_cal_exist)
            if mask3d_dis_cal_exist:
                mask3d_dis_cal[mask3d_dis_noncal>68]=68
                # np.savetxt(os.path.join(dir,mask3d_path+'exist_noncal.txt'), mask3d_dis_noncal,fmt='%d')
            else:
                mask3d_dis_cal = mask3d_dis_noncal + 68
            mask_dis_inner =os.path.join(mask_dis_dir , slice) + '_inner.tiff'
            mask_dis_outer = os.path.join(mask_dis_dir , slice) + '_outer.tiff'
            mask_dis = np.zeros([3,96,96])
            mask_dis[0]=np.loadtxt(mask_dis_inner)
            mask_dis[1] = np.loadtxt(mask_dis_outer)
            # mask_dis[2] = mask3d_dis_noncal
            # mask_dis[3] = mask3d_dis_cal
            for i in image_path_axis.split('/')[:-3]:
                risk_label_path =risk_label_path+i+'/'
            for i in image_path_axis.split('/')[:-1]:
                image_path = image_path + i + '/'
            risk_label_path = risk_label_path+'risk_labels.txt'


            # print("image_path:", image_path)
            # print("risk_label_path:",risk_label_path)
            slice_files = sorted(
                [file for file in listdir(image_path) if file.endswith('.tiff') and not file.startswith('.')])


            start_file, end_file = slice_files[0], slice_files[-1]
            # print('start_file:', start_file)
            # print('end_file:', end_file)

            start_num= int(start_file.split('.tiff')[0])
            end_num = int(end_file.split('.tiff')[0])

            # print("start_num:",int(start_num))
            # sys.exit()


            # risk_num = len(np.loadtxt(risk_label_path))
            # assert risk_num==end_num-start_num+1


            # print("risk_num:",risk_num)
            # print("end_num-start_num:",end_num-start_num)


            # risk_label = [np.loadtxt(risk_label_path)[i-start_num] for i in range(int(slice_idx),int(slice_idx)+cfg.interval*cfg.down_sample, cfg.down_sample) ]

            # risk_label = risk_label[cfg.interval//2]

            # print("risk_label:",risk_label)

            # sys.exit()







            slice_files_axis = [osp.join(image_path_axis.replace(slice_idx+".tiff", "{:03d}".format(i)+".tiff"))
                                for i in range(int(slice_idx),int(slice_idx)+cfg.interval*cfg.down_sample, cfg.down_sample)
                                ]

            label_files_axis = [osp.join(mask_path_axis.replace(slice_idx+".tiff", "{:03d}".format(i)+".tiff"))
                                for i in range(int(slice_idx),int(slice_idx)+cfg.interval*cfg.down_sample, cfg.down_sample)]

            mask_axis = np.stack([io.imread(label_file) for label_file in label_files_axis])



            image_axis = np.stack([io.imread(slice_file) for slice_file in slice_files_axis])
        end1 = time.time()
        if self.split =="train":
            transforms = make_transforms(cfg,"True")
        else:
            transforms = make_transforms(cfg,"False")



        # print()
        sample_img, sample_mask,sample_mask_seg,sample_mask_dis =  transforms([image_axis, mask_axis,mask_axis,mask_dis])
        # print("ok_transform")
        # sample_img, sample_mask, sample_mask_dis = transforms([image_axis, mask_axis, mask_dis])
        # sample_img, sample_mask, sample_mask_seg = transforms([image_axis, mask_axis, mask_axis])
        # print("go")
        # np.savetxt("mask_skip.txt", sample_mask[0],fmt="%d")
        # np.savetxt("maskseg_skip.txt", sample_mask_seg[0], fmt="%d")
        # sample_img, sample_mask, _ = transforms([image_axis, mask_axis, mask_axis])
        sample_mask_f = sample_mask
        sample_mask = sample_mask[cfg.interval//2]
        mask_seg_ids = np.unique(sample_mask_seg)
        if 4 not in mask_seg_ids:
            healthy = 1
        else:
            healthy = 0
        # print("cfg.interval//2:",cfg.interval//2)
        end2 = time.time()
        # sample_img=np.array(anno['annotations']['sample_img'])
        # sample_mask = np.array(anno['annotations']['sample_central_mask'])
        # print("sample_img:",sample_img.shape)


            # if axis_name == 'applicate':
            #     new_d, new_h, new_w = image_axis.shape
            #     image = np.zeros((*image_axis.shape, len(axis_names)), dtype=np.int16)
            #     image[:, :, :, a_inx] = image_axis
        img = np.transpose(sample_img,(1,2,0))
        instance_ids = np.unique(sample_mask)

        polys = []
        annotations=[]
        for instance_id in instance_ids:
            if instance_id == 0 or instance_id == 3 or instance_id == 4:  # background or edge, pass
                continue
            # extract instance

            # temp = np.zeros(sample_mask.shape)
            temp = np.zeros(sample_mask.shape)
            # semantic category of this instance

            temp[sample_mask == instance_id] = 1
            self.fill(temp, (0, 0), 2)
            temp_f = np.ones(sample_mask.shape)
            temp_f[temp == 2] = 0
            # cv2.imwrite("temp_f.png",temp_f*255)
            instance = temp_f

            poly = self.binary_mask_to_polygon(instance)
            # print(poly)
            # print("instance_id:",instance_id)
            if len(poly) == 0:
                continue
            annos = {'segmentation': poly,'category_id':int(instance_id)-1}
            annotations.append(annos)
            # instance_polys.append([np.array(poly).reshape(-1, 2)])
        # instance_polys.append(polys)
        # print( instance_polys)
        # print("instance_polys:")



        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in annotations
                          if not isinstance(obj['segmentation'], dict)]
        instance_polys_1 = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno
                          if not isinstance(obj['segmentation'], dict)]
        # print(instance_polys)
        # print("instance_polys1:")
        # sys.exit()
        # cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in annotations]
        # print("cls_ids:",cls_ids)
        cls_ids = [obj['category_id'] for obj in annotations]
        end3 = time.time()

        # print("time_readdata:",end1-start)
        # print("time_transform:", end2 - end1)
        # print("time_readpoly:", end3 - end2)
        # print("instance_len:",len(instance_polys))
        # assert len(instance_polys)==2
        # print("cls_ids:",cls_ids)
        # return img, instance_polys, cls_ids,sample_mask_f,instance_polys_1,annotations,sample_mask_seg
        return img, instance_polys, cls_ids,sample_mask_f,instance_polys_1,annotations,sample_mask_dis,sample_mask_seg,healthy
        # return img, instance_polys, cls_ids, sample_mask_f, instance_polys_1, annotations, risk_label
        # return img, instance_polys, cls_ids, sample_mask_f, instance_polys_1, annotations
    def transform_original_data(self, instance_polys, flipped, width, trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for py in instance_polys:
            for cood in py:
                for co in cood:
                    assert co[0]<96 and co[1]<96
        for instance in instance_polys:
            polys = instance

            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_
            # print("output_h:",output_h,"width:",output_w)
            # print("trans_output:", trans_output)

            polys = snake_kins_utils.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        # print("after_transform_polys")
        for py in instance_polys:
            for cood in py:
                for co in cood:
                    assert co[0]<96 and co[1]<96
        return instance_polys_

    def get_valid_polys(self, instance_polys):
        instance_polys_ = []
        # print("vallid instance_polys:",instance_polys)

        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            # print( instance,"vallid instance:")
            polys = snake_kins_utils.filter_tiny_polys(instance)
            # print(polys, "tiny polys:")
            polys = snake_kins_utils.get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            polys = [poly for poly in polys if len(poly) >= 4]
            instance_polys_.append(polys)
        return instance_polys_

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = [snake_kins_utils.get_extreme_points(poly) for poly in instance]
            extreme_points.append(points)
        return extreme_points

    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind):
        # print("cls_id:",cls_id)
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct = np.round(ct).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

    def prepare_init(self, box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, h, w):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        img_init_poly = snake_kins_utils.get_init(box)
        img_init_poly = snake_kins_utils.uniformsample(img_init_poly, snake_config.init_poly_num)
        can_init_poly = snake_kins_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = extreme_point
        can_gt_poly = snake_kins_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        i_it_4pys.append(img_init_poly)
        c_it_4pys.append(can_init_poly)
        i_gt_4pys.append(img_gt_poly)
        c_gt_4pys.append(can_gt_poly)

    def prepare_evolution(self, poly, extreme_point, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        # octagon = snake_kins_utils.get_ellipse_contour(poly)
        octagon = snake_kins_utils.get_octagon(extreme_point)
        img_init_poly = snake_kins_utils.uniformsample(octagon, snake_config.poly_num)
        can_init_poly = snake_kins_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)

        img_gt_poly = snake_kins_utils.uniformsample(poly, len(poly) * snake_config.gt_poly_num)
        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = snake_kins_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def prepare_merge(self, is_id, cls_id, cp_id, cp_cls):
        cp_id.append(is_id)
        cp_cls.append(cls_id)
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
    # feature_risk_label


#nc_only
    def __getitem__(self, index):
        # print("index:",index)
        # print("new")
        # print("index:",index)
        start = time.time()
        # print(self.anns_nc)
        # print("len_nc:",len(self.anns_nc))
        # print("index:", index)

        # mixup = np.random.RandomState().rand()
        # print("mixup:",mixup)
        # prob = np.random.RandomState().rand()
        # print("prob:",prob)
        # mix = 1
        # if prob < 1:
        #     mix = 1
        # else:
        #     mix = 0

        # ann_nc_id = np.random.RandomState().randint(len(self.anns_nc))
        # print("ann_nc_id:",ann_nc_id)
        # anno_nc, path_nc, img_id_nc = self.process_info(self.anns_nc[ann_nc_id])

        # sys.exit()
        # for i in range(len(self.anns_nc)):
        #     anno_nc, path_nc, img_id_nc = self.process_info(self.anns_nc[i])
        #     if anno_nc :
        #         break
        # ann_nc = self.anns_nc[ann_nc_id]
        ann = self.anns[index]
        # print("annout:",ann)
        # print("ann_nc:",anno_nc)
        # sys.exit()


        anno, path, img_id = self.process_info(ann)

        # anno_nc, path_nc, img_id_nc = self.process_info(ann_nc)

        # print("in _ batch")


        img, instance_polys, cls_ids, mask, instance_polys_1, annotations, sample_mask_dis,mask_seg_plaque,healthy = self.read_original_data(
            anno, path)
        #
        # print("cls_ids:",cls_ids)
        # print("instance_polys_len:",len(instance_polys))
        # print("instance_polys[0]:",instance_polys[0][0].shape)
        # print("cls_ids_nc:",cls_ids_nc)
        # print("instance_polys_nc_len:",len(instance_polys_nc))
        # print("instance_polys_nc[0]:",instance_polys_nc[0][0].shape)




        # if mix:
        #     img_mix = mixup*img_nc+(1-mixup)*img
        # else:
        #     img_mix = img

        # fig, ax = plt.subplots(3,1, figsize=(20, 20))
        # fig.tight_layout()
        # # ax.axis('off')
        # # print(img[:,:,7].shape)
        # # sys.exit()
        #
        #
        # ax[0].imshow(img[:,:,7], cmap='gray')
        # ax[2].imshow(img_nc[:,:,7], cmap='gray')
        # # ax[2, 0].imshow(inner, cmap='gray')
        # # ax[2, 1].imshow(outer, cmap='gray')
        # # ax[2, 0].imshow(gt_seg_plq[0], cmap='gray')
        # # ax[2, 1].imshow(gt_seg_plq[1], cmap='gray')
        # # ax[2, 1].imshow(output_health[2], cmap='gray')
        # # ax[2, 2].imshow(output_unhealth[2], cmap='gray')
        # ax[1].imshow(img_mix[:,:,7], cmap='gray')
        # fig.tight_layout()
        # ax[0].set_title(path)
        # ax[2].set_title(path_nc)
        # ax[1].set_title(mixup)
        #
        # from os.path import exists
        # from os import mkdir
        # if not exists('mixup'):
        #     mkdir('mixup')
        # plt.savefig("mixup/mixup_{}.png".format(index))








        img_f=img.astype(np.float32)
        img_f = img_f.transpose(2, 0, 1)
        end1 = time.time()

        start1 = time.time()
        height, width = img.shape[0], img.shape[1]

        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_kins_utils.augment(
                img, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                snake_config.mean, snake_config.std, instance_polys
            )

        instance_polys = self.get_valid_polys(instance_polys)

        extreme_points = self.get_extreme_points(instance_polys)

        #
        # instance_polys_nc = self.get_valid_polys(instance_polys_nc)
        #
        # extreme_points_nc = self.get_extreme_points(instance_polys_nc)

        output_h, output_w = inp_out_hw[2:]

        ct_hm = np.zeros([2, output_h, output_w], dtype=np.float32)
        ct_hm_nc = np.zeros([2, output_h, output_w], dtype=np.float32)

        end2= time.time()
        # print("index:",index,"time_poly_prepare:",end2-start1)

        wh = []
        ct_cls = []
        ct_ind = []
        wh_nc = []
        ct_cls_nc = []
        ct_ind_nc = []

        # init
        i_it_4pys = []
        c_it_4pys = []
        i_gt_4pys = []
        c_gt_4pys = []
        # i_it_4pys_nc = []
        # c_it_4pys_nc = []
        # i_gt_4pys_nc = []
        # c_gt_4pys_nc = []

        # evolution
        i_it_pys = []
        c_it_pys = []
        i_gt_pys = []
        c_gt_pys = []
        # i_it_pys_nc = []
        # c_it_pys_nc = []
        # i_gt_pys_nc = []
        # c_gt_pys_nc = []

        # print("anno_len:",len(anno))
        # assert len(anno)==2
        # assert len(instance_polys)==2

        for i in range(len(annotations)):
            cls_id = cls_ids[i]

            instance_poly = instance_polys[i]
            instance_points = extreme_points[i]

            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                if not snake_kins_utils.is_ellipse_valid(poly):
                    print("poly skipped ")
                    continue
                # assert not len(poly)
                extreme_point = instance_points[j]

                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue

                decode_box = self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
                self.prepare_init(decode_box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
                self.prepare_evolution(poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)
        # print("len+len_nc:",len(annotations)+len(annotations_nc))
        # print("i_it_4pys_len:",len(i_it_4pys))
        # print("i_it_4pys[0]:",i_it_4pys[0].shape)
        # print("ct_hm:",ct_hm)
        # sys.exit()

        # for i in range(len(annotations_nc)):
        #     cls_id_nc = cls_ids_nc[i]
        #
        #     instance_poly = instance_polys_nc[i]
        #     instance_points = extreme_points_nc[i]
        #
        #     for j in range(len(instance_poly)):
        #         poly = instance_poly[j]
        #         # assert not len(poly)
        #         extreme_point = instance_points[j]
        #
        #         x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
        #         x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
        #         bbox = [x_min, y_min, x_max, y_max]
        #         h, w = y_max - y_min + 1, x_max - x_min + 1
        #         if h <= 1 or w <= 1:
        #             continue
        #
        #         decode_box = self.prepare_detection(bbox, poly, ct_hm_nc, cls_id_nc, wh_nc, ct_cls_nc, ct_ind_nc)
        #         self.prepare_init(decode_box, extreme_point, i_it_4pys_nc, c_it_4pys_nc, i_gt_4pys_nc, c_gt_4pys_nc, output_h, output_w)
        #         self.prepare_evolution(poly, extreme_point, i_it_pys_nc, c_it_pys_nc, i_gt_pys_nc, c_gt_pys_nc)
        # print('i_gt_py_len_before', len(i_gt_pys))
        # assert len(i_gt_pys)==2
        start2=time.time()
        mask_3d = mask


        # mask_mix = np.zeros([5,96,96])
        # if mix:
        #     mask_mix[0][mask==1]=float(1-mixup)
        #     mask_mix[1][mask==2]=float(1-mixup)
        #     mask_mix[2][mask_nc==1]=float(mixup)
        #     mask_mix[3][mask_nc==2]=float(mixup)
        # else:
        #     if healthy:
        #         mask_mix[0][mask==1]=float(1)
        #         mask_mix[1][mask==2]=float(1)
        #     else:
        #         mask_mix[2][mask==1]=float(1)
        #         mask_mix[3][mask==2]=float(1)



        # from os.path import exists
        # from os import mkdir
        # if not exists('mixup'):
        #     mkdir('mixup')
        # np.savetxt("mixup/mixup_0_{}.txt".format(index),mask_mix[0],fmt='%7.2f')
        # np.savetxt("mixup/mixup_1_{}.txt".format(index),mask_mix[1],fmt='%7.2f')
        # np.savetxt("mixup/mixup_2_{}.txt".format(index),mask_mix[2],fmt='%7.2f')
        # np.savetxt("mixup/mixup_3_{}.txt".format(index),mask_mix[3],fmt='%7.2f')
        #
        # sys.exit()




        mask_seg = np.ones(mask.shape)
        mask_1 = np.ones(mask.shape)
        # 1 for inner mask_1 is segmentation
        mask_back = np.ones(mask.shape)
        mask_wall = np.ones(mask.shape)
        mask_2 = np.ones(mask.shape)

        mask_dis = np.zeros([3,96,96])

        # 2 for outer mask_2 is segmentation

        mask_1[mask == 1] = 0
        mask_2[mask == 2] = 0

        mask_back[mask_2 == 1] = 0
        mask_wall = mask_2 - mask_1

        mask_seg[mask_back == 1] = 0
        # background
        mask_seg[mask_wall == 1] = 1
        #  artery wall
        mask_seg[mask_1 == 1] = 2


        # sys.exit()
        for i in range(len(i_it_pys)):
            i_it_pys[i]=i_it_pys[i]/snake_config.down_ratio
        for i in range(len(c_it_pys)):
            c_it_pys[i]=c_it_pys[i]/snake_config.down_ratio
        for i in range(len(i_gt_pys)):
            i_gt_pys[i]=i_gt_pys[i]/snake_config.down_ratio
        for i in range(len(c_gt_pys)):
            c_gt_pys[i]=c_gt_pys[i]/snake_config.down_ratio

        # for i in range(len(i_it_pys_nc)):
        #     i_it_pys_nc[i]=i_it_pys_nc[i]/snake_config.down_ratio
        # for i in range(len(c_it_pys_nc)):
        #     c_it_pys_nc[i]=c_it_pys_nc[i]/snake_config.down_ratio
        # for i in range(len(i_gt_pys_nc)):
        #     i_gt_pys_nc[i]=i_gt_pys_nc[i]/snake_config.down_ratio
        # for i in range(len(c_gt_pys_nc)):
        #     c_gt_pys_nc[i]=c_gt_pys_nc[i]/snake_config.down_ratio
        thickness_set=np.zeros(snake_config.poly_num)

        ct_health = np.unique(mask_seg_plaque)
        label_health = np.array([1,0,0],dtype=float)
        if 3 in ct_health :
            label_health = np.array([0,1,0],dtype=float)
        if 4 in ct_health:
            label_health = np.array([0, 0, 1], dtype=float)

        # ct_health = np.unique(mask_seg_plaque)
        # label_health = np.array([1,0,0,0],dtype=float)
        # if 3 in ct_health :
        #     label_health = np.array([0,1,0,0],dtype=float)
        # if 4 in ct_health:
        #     label_health = np.array([0, 0, 1,0], dtype=float)
        # if 3 in ct_health and 4 in ct_health:
        #     label_health = np.array([0, 0,0, 1], dtype=float)
        # label_health = np.zeros([15,4],dtype=float)
        # for i in range(len(label_health)):
        #     ct_health = np.unique(mask_seg_plaque[i])
        #     # print("i:",i)
        #     # print(" ct_health:",  ct_health)
        #     label_health[i] = np.array([1, 0, 0, 0], dtype=float)
        #     if 3 in ct_health:
        #         label_health[i] = np.array([0, 1, 0, 0], dtype=float)
        #     if 4 in ct_health:
        #         label_health[i] = np.array([0, 0, 1, 0], dtype=float)
        #     if 3 in ct_health and 4 in ct_health:
        #         label_health[i] = np.array([0, 0, 0, 1], dtype=float)
        # print("label_health:",label_health)
        # sys.exit()




        # ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg, 'path': path,
        #        'mask_seg_plaque': mask_seg_plaque}
        # ret = {'inp': img_f,'poly':instance_polys,'mask':mask,'mask_seg':mask_seg,'path':path,'risk_label':risk_onehot,'thickness_set':thickness_set}

        # ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg, 'path': path, 'mask_dis':sample_mask_dis,'mask_seg_plaque': mask_seg_plaque}

        # print()
        mask_seg_wall = np.zeros(mask_seg_plaque[7].shape)
        mask_seg_plq_only = np.zeros(mask_seg_plaque[7].shape)
        mask_seg_wall[mask_seg_plaque[7]==0]=0
        mask_seg_wall[mask_seg_plaque[7]==1]=0
        mask_seg_wall[mask_seg_plaque[7]==2]=1
        mask_seg_wall[mask_seg_plaque[7]==3]=1
        mask_seg_wall[mask_seg_plaque[7]==4]=1
        # mask_seg_plq_only[mask_seg_plaque[7]==3]=1
        mask_seg_plq_only[mask_seg_plaque[7]==4]=1
        ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask[7], 'mask_seg': mask_seg_plaque[7],'mask_seg_wall':mask_seg_wall,'mask_seg_plq_only':mask_seg_plq_only, 'path': path,
               'mask_dis': sample_mask_dis, 'mask_seg_plaque': mask_seg,'label_health':label_health}
        # print("inp_shape:",img_f.shape)
        # print("poly_shape:", len(instance_polys))
        # print("mask_shape:", mask.shape)
        # print("mask_seg_shape:", mask_seg.shape)
        # ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg, 'path': path}
        # print(path)
        # sys.exit()
        # print("inp_shape:",inp.shape)
        # print("instance_polys_shape:", len(instance_polys))
        # print("mask_shape:", mask.shape)
        # sys.exit()
        # print("len_poly:",len(ret['poly']))

        detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
        evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}

        # from os.path import exists
        # from os import mkdir
        # if not exists('mask_seg_plaque'):
        #     mkdir('mask_seg_plaque')
        # np.savetxt("mask_seg_plaque/mask_seg_plaque_{}.txt".format(index), mask_seg_plaque[7],fmt='%d')
        # np.savetxt("mask_seg_plaque/mask_seg_simp_{}.txt".format(index), mask_seg_wall,fmt='%d')
        # np.savetxt("mask_seg_plaque/mask_seg_plq_only_{}.txt".format(index), mask_seg_plq_only,fmt='%d')

        # print("len_ig_t_pys:", len(i_gt_pys))
        # assert len(i_gt_pys)==2
        # print(i_gt_pys[1])
        t=0
        # for i in i_gt_pys:
        #     with open('py/py_%d.txt'%t, 'w') as outfile:
        #         np.savetxt(outfile, i, fmt='%s')
        #     t+=1
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt





        ret.update(detection)
        ret.update(init)
        ret.update(evolution)
        # visualize_utils.visualize_snake_detection(orig_img, ret)
        # visualize_utils.visualize_snake_evolution(orig_img, ret)

        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'ct_num': ct_num}
        ret.update({'meta': meta})

        # print(ret['poly'])
        end = time.time()
        # print("index:",index,"time_segmask:",end-start2)
        # print("index:",index,"time_all:", end - start)
        # print("----------------------------------------------")
        return ret

# normal
    # def __getitem__(self, index):
    #     # print("index:",index)
    #     # print("new")
    #     # print("index:",index)
    #     start = time.time()
    #     # print(self.anns_nc)
    #     # print("len_nc:",len(self.anns_nc))
    #     # print("index:", index)
    #
    #     # mixup = np.random.RandomState().rand()
    #     # print("mixup:",mixup)
    #     # prob = np.random.RandomState().rand()
    #     # print("prob:",prob)
    #     # mix = 1
    #     # if prob < 1:
    #     #     mix = 1
    #     # else:
    #     #     mix = 0
    #
    #     # ann_nc_id = np.random.RandomState().randint(len(self.anns_nc))
    #     # print("ann_nc_id:",ann_nc_id)
    #     # anno_nc, path_nc, img_id_nc = self.process_info(self.anns_nc[ann_nc_id])
    #
    #     # sys.exit()
    #     # for i in range(len(self.anns_nc)):
    #     #     anno_nc, path_nc, img_id_nc = self.process_info(self.anns_nc[i])
    #     #     if anno_nc :
    #     #         break
    #     # ann_nc = self.anns_nc[ann_nc_id]
    #     ann = self.anns[index]
    #     # print("annout:",ann)
    #     # print("ann_nc:",anno_nc)
    #     # sys.exit()
    #
    #
    #     anno, path, img_id = self.process_info(ann)
    #
    #     # anno_nc, path_nc, img_id_nc = self.process_info(ann_nc)
    #
    #     # print("in _ batch")
    #
    #
    #     img, instance_polys, cls_ids, mask, instance_polys_1, annotations, sample_mask_dis,mask_seg_plaque,healthy = self.read_original_data(
    #         anno, path)
    #     #
    #     # print("cls_ids:",cls_ids)
    #     # print("instance_polys_len:",len(instance_polys))
    #     # print("instance_polys[0]:",instance_polys[0][0].shape)
    #     # print("cls_ids_nc:",cls_ids_nc)
    #     # print("instance_polys_nc_len:",len(instance_polys_nc))
    #     # print("instance_polys_nc[0]:",instance_polys_nc[0][0].shape)
    #
    #
    #
    #
    #     # if mix:
    #     #     img_mix = mixup*img_nc+(1-mixup)*img
    #     # else:
    #     #     img_mix = img
    #
    #     # fig, ax = plt.subplots(3,1, figsize=(20, 20))
    #     # fig.tight_layout()
    #     # # ax.axis('off')
    #     # # print(img[:,:,7].shape)
    #     # # sys.exit()
    #     #
    #     #
    #     # ax[0].imshow(img[:,:,7], cmap='gray')
    #     # ax[2].imshow(img_nc[:,:,7], cmap='gray')
    #     # # ax[2, 0].imshow(inner, cmap='gray')
    #     # # ax[2, 1].imshow(outer, cmap='gray')
    #     # # ax[2, 0].imshow(gt_seg_plq[0], cmap='gray')
    #     # # ax[2, 1].imshow(gt_seg_plq[1], cmap='gray')
    #     # # ax[2, 1].imshow(output_health[2], cmap='gray')
    #     # # ax[2, 2].imshow(output_unhealth[2], cmap='gray')
    #     # ax[1].imshow(img_mix[:,:,7], cmap='gray')
    #     # fig.tight_layout()
    #     # ax[0].set_title(path)
    #     # ax[2].set_title(path_nc)
    #     # ax[1].set_title(mixup)
    #     #
    #     # from os.path import exists
    #     # from os import mkdir
    #     # if not exists('mixup'):
    #     #     mkdir('mixup')
    #     # plt.savefig("mixup/mixup_{}.png".format(index))
    #
    #
    #
    #
    #
    #
    #
    #
    #     img_f=img.astype(np.float32)
    #     img_f = img_f.transpose(2, 0, 1)
    #     end1 = time.time()
    #
    #     start1 = time.time()
    #     height, width = img.shape[0], img.shape[1]
    #
    #     orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
    #         snake_kins_utils.augment(
    #             img, self.split,
    #             snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
    #             snake_config.mean, snake_config.std, instance_polys
    #         )
    #
    #     instance_polys = self.get_valid_polys(instance_polys)
    #
    #     extreme_points = self.get_extreme_points(instance_polys)
    #
    #     #
    #     # instance_polys_nc = self.get_valid_polys(instance_polys_nc)
    #     #
    #     # extreme_points_nc = self.get_extreme_points(instance_polys_nc)
    #
    #     output_h, output_w = inp_out_hw[2:]
    #
    #     ct_hm = np.zeros([2, output_h, output_w], dtype=np.float32)
    #     ct_hm_nc = np.zeros([2, output_h, output_w], dtype=np.float32)
    #
    #     end2= time.time()
    #     # print("index:",index,"time_poly_prepare:",end2-start1)
    #
    #     wh = []
    #     ct_cls = []
    #     ct_ind = []
    #     wh_nc = []
    #     ct_cls_nc = []
    #     ct_ind_nc = []
    #
    #     # init
    #     i_it_4pys = []
    #     c_it_4pys = []
    #     i_gt_4pys = []
    #     c_gt_4pys = []
    #     # i_it_4pys_nc = []
    #     # c_it_4pys_nc = []
    #     # i_gt_4pys_nc = []
    #     # c_gt_4pys_nc = []
    #
    #     # evolution
    #     i_it_pys = []
    #     c_it_pys = []
    #     i_gt_pys = []
    #     c_gt_pys = []
    #     # i_it_pys_nc = []
    #     # c_it_pys_nc = []
    #     # i_gt_pys_nc = []
    #     # c_gt_pys_nc = []
    #
    #     # print("anno_len:",len(anno))
    #     # assert len(anno)==2
    #     # assert len(instance_polys)==2
    #
    #     for i in range(len(annotations)):
    #         cls_id = cls_ids[i]
    #
    #         instance_poly = instance_polys[i]
    #         instance_points = extreme_points[i]
    #
    #         for j in range(len(instance_poly)):
    #             poly = instance_poly[j]
    #             # assert not len(poly)
    #             extreme_point = instance_points[j]
    #
    #             x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
    #             x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
    #             bbox = [x_min, y_min, x_max, y_max]
    #             h, w = y_max - y_min + 1, x_max - x_min + 1
    #             if h <= 1 or w <= 1:
    #                 continue
    #
    #             decode_box = self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
    #             self.prepare_init(decode_box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
    #             self.prepare_evolution(poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)
    #     # print("len+len_nc:",len(annotations)+len(annotations_nc))
    #     # print("i_it_4pys_len:",len(i_it_4pys))
    #     # print("i_it_4pys[0]:",i_it_4pys[0].shape)
    #     # print("ct_hm:",ct_hm)
    #     # sys.exit()
    #
    #     # for i in range(len(annotations_nc)):
    #     #     cls_id_nc = cls_ids_nc[i]
    #     #
    #     #     instance_poly = instance_polys_nc[i]
    #     #     instance_points = extreme_points_nc[i]
    #     #
    #     #     for j in range(len(instance_poly)):
    #     #         poly = instance_poly[j]
    #     #         # assert not len(poly)
    #     #         extreme_point = instance_points[j]
    #     #
    #     #         x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
    #     #         x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
    #     #         bbox = [x_min, y_min, x_max, y_max]
    #     #         h, w = y_max - y_min + 1, x_max - x_min + 1
    #     #         if h <= 1 or w <= 1:
    #     #             continue
    #     #
    #     #         decode_box = self.prepare_detection(bbox, poly, ct_hm_nc, cls_id_nc, wh_nc, ct_cls_nc, ct_ind_nc)
    #     #         self.prepare_init(decode_box, extreme_point, i_it_4pys_nc, c_it_4pys_nc, i_gt_4pys_nc, c_gt_4pys_nc, output_h, output_w)
    #     #         self.prepare_evolution(poly, extreme_point, i_it_pys_nc, c_it_pys_nc, i_gt_pys_nc, c_gt_pys_nc)
    #     # print('i_gt_py_len_before', len(i_gt_pys))
    #     # assert len(i_gt_pys)==2
    #     start2=time.time()
    #     mask_3d = mask
    #
    #
    #     # mask_mix = np.zeros([5,96,96])
    #     # if mix:
    #     #     mask_mix[0][mask==1]=float(1-mixup)
    #     #     mask_mix[1][mask==2]=float(1-mixup)
    #     #     mask_mix[2][mask_nc==1]=float(mixup)
    #     #     mask_mix[3][mask_nc==2]=float(mixup)
    #     # else:
    #     #     if healthy:
    #     #         mask_mix[0][mask==1]=float(1)
    #     #         mask_mix[1][mask==2]=float(1)
    #     #     else:
    #     #         mask_mix[2][mask==1]=float(1)
    #     #         mask_mix[3][mask==2]=float(1)
    #
    #
    #
    #     # from os.path import exists
    #     # from os import mkdir
    #     # if not exists('mixup'):
    #     #     mkdir('mixup')
    #     # np.savetxt("mixup/mixup_0_{}.txt".format(index),mask_mix[0],fmt='%7.2f')
    #     # np.savetxt("mixup/mixup_1_{}.txt".format(index),mask_mix[1],fmt='%7.2f')
    #     # np.savetxt("mixup/mixup_2_{}.txt".format(index),mask_mix[2],fmt='%7.2f')
    #     # np.savetxt("mixup/mixup_3_{}.txt".format(index),mask_mix[3],fmt='%7.2f')
    #     #
    #     # sys.exit()
    #
    #
    #
    #
    #     mask_seg = np.ones(mask.shape)
    #     mask_1 = np.ones(mask.shape)
    #     # 1 for inner mask_1 is segmentation
    #     mask_back = np.ones(mask.shape)
    #     mask_wall = np.ones(mask.shape)
    #     mask_2 = np.ones(mask.shape)
    #
    #     mask_dis = np.zeros([3,96,96])
    #
    #     # 2 for outer mask_2 is segmentation
    #
    #     mask_1[mask == 1] = 0
    #     mask_2[mask == 2] = 0
    #
    #     mask_back[mask_2 == 1] = 0
    #     mask_wall = mask_2 - mask_1
    #
    #     mask_seg[mask_back == 1] = 0
    #     # background
    #     mask_seg[mask_wall == 1] = 1
    #     #  artery wall
    #     mask_seg[mask_1 == 1] = 2
    #
    #
    #     # sys.exit()
    #     for i in range(len(i_it_pys)):
    #         i_it_pys[i]=i_it_pys[i]/snake_config.down_ratio
    #     for i in range(len(c_it_pys)):
    #         c_it_pys[i]=c_it_pys[i]/snake_config.down_ratio
    #     for i in range(len(i_gt_pys)):
    #         i_gt_pys[i]=i_gt_pys[i]/snake_config.down_ratio
    #     for i in range(len(c_gt_pys)):
    #         c_gt_pys[i]=c_gt_pys[i]/snake_config.down_ratio
    #
    #     # for i in range(len(i_it_pys_nc)):
    #     #     i_it_pys_nc[i]=i_it_pys_nc[i]/snake_config.down_ratio
    #     # for i in range(len(c_it_pys_nc)):
    #     #     c_it_pys_nc[i]=c_it_pys_nc[i]/snake_config.down_ratio
    #     # for i in range(len(i_gt_pys_nc)):
    #     #     i_gt_pys_nc[i]=i_gt_pys_nc[i]/snake_config.down_ratio
    #     # for i in range(len(c_gt_pys_nc)):
    #     #     c_gt_pys_nc[i]=c_gt_pys_nc[i]/snake_config.down_ratio
    #     thickness_set=np.zeros(snake_config.poly_num)
    #
    #     ct_health = np.unique(mask_seg_plaque)
    #     label_health = np.array([1,0,0],dtype=float)
    #     if 3 in ct_health :
    #         label_health = np.array([0,1,0],dtype=float)
    #     if 4 in ct_health:
    #         label_health = np.array([0, 0, 1], dtype=float)
    #
    #     # ct_health = np.unique(mask_seg_plaque)
    #     # label_health = np.array([1,0,0,0],dtype=float)
    #     # if 3 in ct_health :
    #     #     label_health = np.array([0,1,0,0],dtype=float)
    #     # if 4 in ct_health:
    #     #     label_health = np.array([0, 0, 1,0], dtype=float)
    #     # if 3 in ct_health and 4 in ct_health:
    #     #     label_health = np.array([0, 0,0, 1], dtype=float)
    #     # label_health = np.zeros([15,4],dtype=float)
    #     # for i in range(len(label_health)):
    #     #     ct_health = np.unique(mask_seg_plaque[i])
    #     #     # print("i:",i)
    #     #     # print(" ct_health:",  ct_health)
    #     #     label_health[i] = np.array([1, 0, 0, 0], dtype=float)
    #     #     if 3 in ct_health:
    #     #         label_health[i] = np.array([0, 1, 0, 0], dtype=float)
    #     #     if 4 in ct_health:
    #     #         label_health[i] = np.array([0, 0, 1, 0], dtype=float)
    #     #     if 3 in ct_health and 4 in ct_health:
    #     #         label_health[i] = np.array([0, 0, 0, 1], dtype=float)
    #     # print("label_health:",label_health)
    #     # sys.exit()
    #
    #
    #
    #
    #     # ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg, 'path': path,
    #     #        'mask_seg_plaque': mask_seg_plaque}
    #     # ret = {'inp': img_f,'poly':instance_polys,'mask':mask,'mask_seg':mask_seg,'path':path,'risk_label':risk_onehot,'thickness_set':thickness_set}
    #
    #     # ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg, 'path': path, 'mask_dis':sample_mask_dis,'mask_seg_plaque': mask_seg_plaque}
    #
    #     print()
    #     ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask[7], 'mask_seg': mask_seg_plaque[7], 'path': path,
    #            'mask_dis': sample_mask_dis, 'mask_seg_plaque': mask_seg,'label_health':label_health}
    #     # print("inp_shape:",img_f.shape)
    #     # print("poly_shape:", len(instance_polys))
    #     # print("mask_shape:", mask.shape)
    #     # print("mask_seg_shape:", mask_seg.shape)
    #     # ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg, 'path': path}
    #     # print(path)
    #     # sys.exit()
    #     # print("inp_shape:",inp.shape)
    #     # print("instance_polys_shape:", len(instance_polys))
    #     # print("mask_shape:", mask.shape)
    #     # sys.exit()
    #     # print("len_poly:",len(ret['poly']))
    #
    #     detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
    #     init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
    #     evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
    #     # print("len_ig_t_pys:", len(i_gt_pys))
    #     # assert len(i_gt_pys)==2
    #     # print(i_gt_pys[1])
    #     t=0
    #     # for i in i_gt_pys:
    #     #     with open('py/py_%d.txt'%t, 'w') as outfile:
    #     #         np.savetxt(outfile, i, fmt='%s')
    #     #     t+=1
    #     # I'm writing a header here just for the sake of readability
    #     # Any line starting with "#" will be ignored by numpy.loadtxt
    #
    #
    #
    #
    #
    #     ret.update(detection)
    #     ret.update(init)
    #     ret.update(evolution)
    #     # visualize_utils.visualize_snake_detection(orig_img, ret)
    #     # visualize_utils.visualize_snake_evolution(orig_img, ret)
    #
    #     ct_num = len(ct_ind)
    #     meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'ct_num': ct_num}
    #     ret.update({'meta': meta})
    #
    #     # print(ret['poly'])
    #     end = time.time()
    #     # print("index:",index,"time_segmask:",end-start2)
    #     # print("index:",index,"time_all:", end - start)
    #     # print("----------------------------------------------")
    #     return ret

# # mix_up
#     def __getitem__(self, index):
#         # print("index:",index)
#         # print("new")
#         # print("index:",index)
#         start = time.time()
#         # print(self.anns_nc)
#         # print("len_nc:",len(self.anns_nc))
#         # print("index:", index)
#
#         mixup = np.random.RandomState().rand()
#         # print("mixup:",mixup)
#         prob = np.random.RandomState().rand()
#         # print("prob:",prob)
#         mix = 1
#         # if prob < 1:
#         #     mix = 1
#         # else:
#         #     mix = 0
#
#         ann_nc_id = np.random.RandomState().randint(len(self.anns_nc))
#         # print("ann_nc_id:",ann_nc_id)
#         anno_nc, path_nc, img_id_nc = self.process_info(self.anns_nc[ann_nc_id])
#
#         # sys.exit()
#         # for i in range(len(self.anns_nc)):
#         #     anno_nc, path_nc, img_id_nc = self.process_info(self.anns_nc[i])
#         #     if anno_nc :
#         #         break
#         # ann_nc = self.anns_nc[ann_nc_id]
#         ann = self.anns[index]
#         # print("annout:",ann)
#         # print("ann_nc:",anno_nc)
#         # sys.exit()
#
#
#         anno, path, img_id = self.process_info(ann)
#
#         # anno_nc, path_nc, img_id_nc = self.process_info(ann_nc)
#
#         # print("in _ batch")
#         img_nc, instance_polys_nc, cls_ids_nc, mask_nc, instance_polys_1_nc, annotations_nc, sample_mask_dis_nc,mask_seg_plaque_nc,_ = self.read_original_data(
#             anno_nc, path_nc)
#
#         img, instance_polys, cls_ids, mask, instance_polys_1, annotations, sample_mask_dis,mask_seg_plaque,healthy = self.read_original_data(
#             anno, path)
#         #
#         # print("cls_ids:",cls_ids)
#         # print("instance_polys_len:",len(instance_polys))
#         # print("instance_polys[0]:",instance_polys[0][0].shape)
#         # print("cls_ids_nc:",cls_ids_nc)
#         # print("instance_polys_nc_len:",len(instance_polys_nc))
#         # print("instance_polys_nc[0]:",instance_polys_nc[0][0].shape)
#         if not healthy:
#             mix = 0
#         if mix:
#             for i in range(len(cls_ids_nc)):
#                 cls_ids.append(cls_ids_nc[i])
#             for i in range(len(instance_polys_nc)):
#                 instance_polys.append((instance_polys_nc[i]))
#         mask= mask[7]
#         mask_nc = mask_nc[7]
#         # mask_seg_plaque = mask_seg_plaque[7]
#
#         mask_seg = mask_seg_plaque[7]
#         mask_seg_nc = mask_seg_plaque_nc[7]
#         mask_multi = np.zeros([5,96,96])
#         mask_nc_multi = np.zeros([5,96,96])
#         mask_seg_multi = np.zeros([9,96,96])
#         mask_seg_nc_multi = np.zeros([9,96,96])
#         if self.split =="test":
#             mix = 0
#         if mix:
#
#
#             mask_multi[3][mask==1]=float(1)
#             mask_multi[4][mask==2]=float(1)
#             mask_seg_multi[5][mask_seg==1] = float(1)
#             mask_seg_multi[6][mask_seg==2] = float(1)
#             mask_seg_multi[7][mask_seg==3] = float(1)
#             mask_seg_multi[8][mask_seg==4] = float(1)
#         else:
#             mask_multi[0][mask==0]=float(1)
#             mask_multi[1][mask==1]=float(1)
#             mask_multi[2][mask==2]=float(1)
#             # mask_seg_multi[0][mask_seg==0] = float(1)
#             # mask_seg_multi[1][mask_seg==1] = float(1)
#             # mask_seg_multi[2][mask_seg==2] = float(1)
#             # mask_seg_multi[3][mask_seg==3] = float(1)
#             # mask_seg_multi[4][mask_seg==4] = float(1)
#
#
#         # mask_nc_multi[0][mask_nc==0]=float(1)
#         if not mix:
#             mask_nc_multi[3][mask_nc==1]=float(1)
#             mask_nc_multi[4][mask_nc==2]=float(1)
#         # mask_seg_nc_multi[5][mask_seg_nc==1] = float(1)
#         # mask_seg_nc_multi[6][mask_seg_nc==2] = float(1)
#         # mask_seg_nc_multi[7][mask_seg_nc==3] = float(1)
#         # mask_seg_nc_multi[8][mask_seg_nc==4] = float(1)
#
#
#
#
#
#
#         # print("after_cls_ids:",cls_ids)
#         # print("after_instance_polys_len:",len(instance_polys))
#         # print("after_instance_polys[0]:",instance_polys[0][0].shape)
#         # print("after_instance_polys[1]:",instance_polys[1][0].shape)
#         # print("after_instance_polys[2]:",instance_polys[2][0].shape)
#         # print("after_instance_polys[3]:",instance_polys[3][0].shape)
#         # print("is_training:",self.split)
#
#         # sys.exit()
#
#
#
#         if mix:
#             img_mix = mixup*img_nc+(1-mixup)*img
#             mask_mix_temp = mixup*mask_nc_multi + (1-mixup)*mask_multi
#             mask_seg_mix_temp = mixup*mask_seg_nc_multi + (1-mixup)*mask_seg_multi
#
#         else:
#             img_mix = img
#             mask_mix_temp = mask_multi
#             mask_seg_mix_temp = mask_seg_multi
#
#
#         # if mix:
#         #     img_mix = mixup*img_nc+(1-mixup)*img
#         # else:
#         #     img_mix = img
#
#         # fig, ax = plt.subplots(3,1, figsize=(20, 20))
#         # fig.tight_layout()
#         # # ax.axis('off')
#         # # print(img[:,:,7].shape)
#         # # sys.exit()
#         #
#         #
#         # ax[0].imshow(img[:,:,7], cmap='gray')
#         # ax[2].imshow(img_nc[:,:,7], cmap='gray')
#         # # ax[2, 0].imshow(inner, cmap='gray')
#         # # ax[2, 1].imshow(outer, cmap='gray')
#         # # ax[2, 0].imshow(gt_seg_plq[0], cmap='gray')
#         # # ax[2, 1].imshow(gt_seg_plq[1], cmap='gray')
#         # # ax[2, 1].imshow(output_health[2], cmap='gray')
#         # # ax[2, 2].imshow(output_unhealth[2], cmap='gray')
#         # ax[1].imshow(img_mix[:,:,7], cmap='gray')
#         # fig.tight_layout()
#         # ax[0].set_title(path)
#         # ax[2].set_title(path_nc)
#         # ax[1].set_title(mixup)
#         #
#         # from os.path import exists
#         # from os import mkdir
#         # if not exists('mixup'):
#         #     mkdir('mixup')
#         # plt.savefig("mixup/mixup_{}.png".format(index))
#
#
#
#
#
#
#
#
#         img_f=img_mix.astype(np.float32)
#         img_f = img_f.transpose(2, 0, 1)
#         end1 = time.time()
#
#         start1 = time.time()
#         height, width = img.shape[0], img.shape[1]
#
#         orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
#             snake_kins_utils.augment(
#                 img, self.split,
#                 snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
#                 snake_config.mean, snake_config.std, instance_polys
#             )
#
#         instance_polys = self.get_valid_polys(instance_polys)
#
#         extreme_points = self.get_extreme_points(instance_polys)
#
#         #
#         # instance_polys_nc = self.get_valid_polys(instance_polys_nc)
#         #
#         # extreme_points_nc = self.get_extreme_points(instance_polys_nc)
#
#         output_h, output_w = inp_out_hw[2:]
#
#         ct_hm = np.zeros([2, output_h, output_w], dtype=np.float32)
#         ct_hm_nc = np.zeros([2, output_h, output_w], dtype=np.float32)
#
#         end2= time.time()
#         # print("index:",index,"time_poly_prepare:",end2-start1)
#
#         wh = []
#         ct_cls = []
#         ct_ind = []
#         wh_nc = []
#         ct_cls_nc = []
#         ct_ind_nc = []
#
#         # init
#         i_it_4pys = []
#         c_it_4pys = []
#         i_gt_4pys = []
#         c_gt_4pys = []
#         # i_it_4pys_nc = []
#         # c_it_4pys_nc = []
#         # i_gt_4pys_nc = []
#         # c_gt_4pys_nc = []
#
#         # evolution
#         i_it_pys = []
#         c_it_pys = []
#         i_gt_pys = []
#         c_gt_pys = []
#         # i_it_pys_nc = []
#         # c_it_pys_nc = []
#         # i_gt_pys_nc = []
#         # c_gt_pys_nc = []
#
#         # print("anno_len:",len(anno))
#         # assert len(anno)==2
#         # assert len(instance_polys)==2
#
#         for i in range(len(annotations)+len(annotations_nc) if mix else len(annotations)):
#             cls_id = cls_ids[i]
#
#             instance_poly = instance_polys[i]
#             instance_points = extreme_points[i]
#
#             for j in range(len(instance_poly)):
#                 poly = instance_poly[j]
#                 # assert not len(poly)
#                 extreme_point = instance_points[j]
#
#                 x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
#                 x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
#                 bbox = [x_min, y_min, x_max, y_max]
#                 h, w = y_max - y_min + 1, x_max - x_min + 1
#                 if h <= 1 or w <= 1:
#                     continue
#
#                 decode_box = self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
#                 self.prepare_init(decode_box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
#                 self.prepare_evolution(poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)
#         # print("len+len_nc:",len(annotations)+len(annotations_nc))
#         # print("i_it_4pys_len:",len(i_it_4pys))
#         # print("i_it_4pys[0]:",i_it_4pys[0].shape)
#         # print("ct_hm:",ct_hm)
#         # sys.exit()
#
#         # for i in range(len(annotations_nc)):
#         #     cls_id_nc = cls_ids_nc[i]
#         #
#         #     instance_poly = instance_polys_nc[i]
#         #     instance_points = extreme_points_nc[i]
#         #
#         #     for j in range(len(instance_poly)):
#         #         poly = instance_poly[j]
#         #         # assert not len(poly)
#         #         extreme_point = instance_points[j]
#         #
#         #         x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
#         #         x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
#         #         bbox = [x_min, y_min, x_max, y_max]
#         #         h, w = y_max - y_min + 1, x_max - x_min + 1
#         #         if h <= 1 or w <= 1:
#         #             continue
#         #
#         #         decode_box = self.prepare_detection(bbox, poly, ct_hm_nc, cls_id_nc, wh_nc, ct_cls_nc, ct_ind_nc)
#         #         self.prepare_init(decode_box, extreme_point, i_it_4pys_nc, c_it_4pys_nc, i_gt_4pys_nc, c_gt_4pys_nc, output_h, output_w)
#         #         self.prepare_evolution(poly, extreme_point, i_it_pys_nc, c_it_pys_nc, i_gt_pys_nc, c_gt_pys_nc)
#         # print('i_gt_py_len_before', len(i_gt_pys))
#         # assert len(i_gt_pys)==2
#         start2=time.time()
#         mask_3d = mask
#
#
#         # mask_mix = np.zeros([5,96,96])
#         # if mix:
#         #     mask_mix[0][mask==1]=float(1-mixup)
#         #     mask_mix[1][mask==2]=float(1-mixup)
#         #     mask_mix[2][mask_nc==1]=float(mixup)
#         #     mask_mix[3][mask_nc==2]=float(mixup)
#         # else:
#         #     if healthy:
#         #         mask_mix[0][mask==1]=float(1)
#         #         mask_mix[1][mask==2]=float(1)
#         #     else:
#         #         mask_mix[2][mask==1]=float(1)
#         #         mask_mix[3][mask==2]=float(1)
#         if self.split =="test":
#             mask_mix = mask_nc_multi
#
#
#
#         # from os.path import exists
#         # from os import mkdir
#         # if not exists('mixup'):
#         #     mkdir('mixup')
#         # np.savetxt("mixup/mixup_0_{}.txt".format(index),mask_mix[0],fmt='%7.2f')
#         # np.savetxt("mixup/mixup_1_{}.txt".format(index),mask_mix[1],fmt='%7.2f')
#         # np.savetxt("mixup/mixup_2_{}.txt".format(index),mask_mix[2],fmt='%7.2f')
#         # np.savetxt("mixup/mixup_3_{}.txt".format(index),mask_mix[3],fmt='%7.2f')
#         #
#         # sys.exit()
#
#
#
#
#         mask_seg = np.ones(mask.shape)
#         mask_1 = np.ones(mask.shape)
#         # 1 for inner mask_1 is segmentation
#         mask_back = np.ones(mask.shape)
#         mask_wall = np.ones(mask.shape)
#         mask_2 = np.ones(mask.shape)
#
#         mask_dis = np.zeros([3,96,96])
#
#         # 2 for outer mask_2 is segmentation
#
#         mask_1[mask == 1] = 0
#         mask_2[mask == 2] = 0
#
#         mask_back[mask_2 == 1] = 0
#         mask_wall = mask_2 - mask_1
#
#         mask_seg[mask_back == 1] = 0
#         # background
#         mask_seg[mask_wall == 1] = 1
#         #  artery wall
#         mask_seg[mask_1 == 1] = 2
#
#
#         # sys.exit()
#         for i in range(len(i_it_pys)):
#             i_it_pys[i]=i_it_pys[i]/snake_config.down_ratio
#         for i in range(len(c_it_pys)):
#             c_it_pys[i]=c_it_pys[i]/snake_config.down_ratio
#         for i in range(len(i_gt_pys)):
#             i_gt_pys[i]=i_gt_pys[i]/snake_config.down_ratio
#         for i in range(len(c_gt_pys)):
#             c_gt_pys[i]=c_gt_pys[i]/snake_config.down_ratio
#
#         # for i in range(len(i_it_pys_nc)):
#         #     i_it_pys_nc[i]=i_it_pys_nc[i]/snake_config.down_ratio
#         # for i in range(len(c_it_pys_nc)):
#         #     c_it_pys_nc[i]=c_it_pys_nc[i]/snake_config.down_ratio
#         # for i in range(len(i_gt_pys_nc)):
#         #     i_gt_pys_nc[i]=i_gt_pys_nc[i]/snake_config.down_ratio
#         # for i in range(len(c_gt_pys_nc)):
#         #     c_gt_pys_nc[i]=c_gt_pys_nc[i]/snake_config.down_ratio
#         thickness_set=np.zeros(snake_config.poly_num)
#
#         ct_health = np.unique(mask_seg_plaque)
#         label_health = np.array([1,0,0],dtype=float)
#         if 3 in ct_health :
#             label_health = np.array([0,1,0],dtype=float)
#         if 4 in ct_health:
#             label_health = np.array([0, 0, 1], dtype=float)
#
#         # ct_health = np.unique(mask_seg_plaque)
#         # label_health = np.array([1,0,0,0],dtype=float)
#         # if 3 in ct_health :
#         #     label_health = np.array([0,1,0,0],dtype=float)
#         # if 4 in ct_health:
#         #     label_health = np.array([0, 0, 1,0], dtype=float)
#         # if 3 in ct_health and 4 in ct_health:
#         #     label_health = np.array([0, 0,0, 1], dtype=float)
#         # label_health = np.zeros([15,4],dtype=float)
#         # for i in range(len(label_health)):
#         #     ct_health = np.unique(mask_seg_plaque[i])
#         #     # print("i:",i)
#         #     # print(" ct_health:",  ct_health)
#         #     label_health[i] = np.array([1, 0, 0, 0], dtype=float)
#         #     if 3 in ct_health:
#         #         label_health[i] = np.array([0, 1, 0, 0], dtype=float)
#         #     if 4 in ct_health:
#         #         label_health[i] = np.array([0, 0, 1, 0], dtype=float)
#         #     if 3 in ct_health and 4 in ct_health:
#         #         label_health[i] = np.array([0, 0, 0, 1], dtype=float)
#         # print("label_health:",label_health)
#         # sys.exit()
#
#
#
#
#         # ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg, 'path': path,
#         #        'mask_seg_plaque': mask_seg_plaque}
#         # ret = {'inp': img_f,'poly':instance_polys,'mask':mask,'mask_seg':mask_seg,'path':path,'risk_label':risk_onehot,'thickness_set':thickness_set}
#         # ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg, 'path': path, 'mask_dis':sample_mask_dis,'mask_seg_plaque': mask_seg_plaque}
#         ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask_mix_temp, 'mask_seg': mask_seg_plaque[7], 'path': path,
#                'mask_dis': sample_mask_dis, 'mask_seg_plaque': mask_seg_mix_temp,'label_health':label_health}
#         # print("inp_shape:",img_f.shape)
#         # print("poly_shape:", len(instance_polys))
#         # print("mask_shape:", mask.shape)
#         # print("mask_seg_shape:", mask_seg.shape)
#         # ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg, 'path': path}
#         # print(path)
#         # sys.exit()
#         # print("inp_shape:",inp.shape)
#         # print("instance_polys_shape:", len(instance_polys))
#         # print("mask_shape:", mask.shape)
#         # sys.exit()
#         # print("len_poly:",len(ret['poly']))
#
#         detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
#         init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
#         evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
#         # print("len_ig_t_pys:", len(i_gt_pys))
#         # assert len(i_gt_pys)==2
#         # print(i_gt_pys[1])
#         t=0
#         # for i in i_gt_pys:
#         #     with open('py/py_%d.txt'%t, 'w') as outfile:
#         #         np.savetxt(outfile, i, fmt='%s')
#         #     t+=1
#         # I'm writing a header here just for the sake of readability
#         # Any line starting with "#" will be ignored by numpy.loadtxt
#
#
#
#
#
#         ret.update(detection)
#         ret.update(init)
#         ret.update(evolution)
#         # visualize_utils.visualize_snake_detection(orig_img, ret)
#         # visualize_utils.visualize_snake_evolution(orig_img, ret)
#
#         ct_num = len(ct_ind)
#         meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'ct_num': ct_num}
#         ret.update({'meta': meta})
#
#         # print(ret['poly'])
#         end = time.time()
#         # print("index:",index,"time_segmask:",end-start2)
#         # print("index:",index,"time_all:", end - start)
#         # print("----------------------------------------------")
#         return ret

    # def __getitem__(self, index):
    #     # print("index:",index)
    #     # print("new")
    #     # print("index:",index)
    #     start = time.time()
    #     # print(self.anns_nc)
    #     # print("len_nc:",len(self.anns_nc))
    #     # print("index:", index)
    #
    #     mixup = np.random.RandomState().rand()
    #     # print("mixup:",mixup)
    #     prob = np.random.RandomState().rand()
    #     # print("prob:",prob)
    #     if prob < 0.4:
    #         mix = 1
    #     else:
    #         mix = 0
    #
    #     ann_nc_id = np.random.RandomState().randint(len(self.anns_nc))
    #     # print("ann_nc_id:",ann_nc_id)
    #     anno_nc, path_nc, img_id_nc = self.process_info(self.anns_nc[ann_nc_id])
    #
    #     # sys.exit()
    #     # for i in range(len(self.anns_nc)):
    #     #     anno_nc, path_nc, img_id_nc = self.process_info(self.anns_nc[i])
    #     #     if anno_nc :
    #     #         break
    #     # ann_nc = self.anns_nc[ann_nc_id]
    #     ann = self.anns[index]
    #     # print("annout:",ann)
    #     # print("ann_nc:",anno_nc)
    #     # sys.exit()
    #
    #
    #     anno, path, img_id = self.process_info(ann)
    #
    #     # anno_nc, path_nc, img_id_nc = self.process_info(ann_nc)
    #
    #     # print("in _ batch")
    #     img_nc, instance_polys_nc, cls_ids_nc, mask_nc, instance_polys_1_nc, annotations_nc, sample_mask_dis_nc,mask_seg_plaque_nc,_ = self.read_original_data(
    #         anno_nc, path_nc)
    #
    #     img, instance_polys, cls_ids, mask, instance_polys_1, annotations, sample_mask_dis,mask_seg_plaque,healthy = self.read_original_data(
    #         anno, path)
    #     #
    #     # print("cls_ids:",cls_ids)
    #     # print("instance_polys_len:",len(instance_polys))
    #     # print("instance_polys[0]:",instance_polys[0][0].shape)
    #     # print("cls_ids_nc:",cls_ids_nc)
    #     # print("instance_polys_nc_len:",len(instance_polys_nc))
    #     # print("instance_polys_nc[0]:",instance_polys_nc[0][0].shape)
    #     if not healthy:
    #         mix = 0
    #     if mix:
    #         for i in range(len(cls_ids_nc)):
    #             cls_ids.append(cls_ids_nc[i])
    #         for i in range(len(instance_polys_nc)):
    #             instance_polys.append((instance_polys_nc[i]))
    #     mask= mask[7]
    #     mask_nc = mask_nc[7]
    #     # mask_seg_plaque = mask_seg_plaque[7]
    #
    #     mask_seg = mask_seg_plaque[7]
    #     mask_seg_nc = mask_seg_plaque_nc[7]
    #     mask_multi = np.zeros([5,96,96])
    #     mask_nc_multi = np.zeros([5,96,96])
    #     mask_seg_multi = np.zeros([9,96,96])
    #     mask_seg_nc_multi = np.zeros([9,96,96])
    #     if self.split =="test":
    #         mix = 0
    #     if mix:
    #         mask_multi[3][mask==1]=float(1)
    #         mask_multi[4][mask==2]=float(1)
    #         mask_seg_multi[5][mask_seg==1] = float(1)
    #         mask_seg_multi[6][mask_seg==2] = float(1)
    #         mask_seg_multi[7][mask_seg==3] = float(1)
    #         mask_seg_multi[8][mask_seg==4] = float(1)
    #     else:
    #         mask_multi[0][mask==0]=float(1)
    #         mask_multi[1][mask==1]=float(1)
    #         mask_multi[2][mask==2]=float(1)
    #         # mask_seg_multi[0][mask_seg==0] = float(1)
    #         # mask_seg_multi[1][mask_seg==1] = float(1)
    #         # mask_seg_multi[2][mask_seg==2] = float(1)
    #         # mask_seg_multi[3][mask_seg==3] = float(1)
    #         # mask_seg_multi[4][mask_seg==4] = float(1)
    #
    #
    #     # mask_nc_multi[0][mask_nc==0]=float(1)
    #     if not mix:
    #         mask_nc_multi[3][mask_nc==1]=float(1)
    #         mask_nc_multi[4][mask_nc==2]=float(1)
    #     # mask_seg_nc_multi[5][mask_seg_nc==1] = float(1)
    #     # mask_seg_nc_multi[6][mask_seg_nc==2] = float(1)
    #     # mask_seg_nc_multi[7][mask_seg_nc==3] = float(1)
    #     # mask_seg_nc_multi[8][mask_seg_nc==4] = float(1)
    #
    #
    #
    #
    #
    #
    #     # print("after_cls_ids:",cls_ids)
    #     # print("after_instance_polys_len:",len(instance_polys))
    #     # print("after_instance_polys[0]:",instance_polys[0][0].shape)
    #     # print("after_instance_polys[1]:",instance_polys[1][0].shape)
    #     # print("after_instance_polys[2]:",instance_polys[2][0].shape)
    #     # print("after_instance_polys[3]:",instance_polys[3][0].shape)
    #     # print("is_training:",self.split)
    #
    #     # sys.exit()
    #
    #
    #
    #     if mix:
    #         img_mix = mixup*img_nc+(1-mixup)*img
    #         mask_mix_temp = mixup*mask_nc_multi + (1-mixup)*mask_multi
    #         mask_seg_mix_temp = mixup*mask_seg_nc_multi + (1-mixup)*mask_seg_multi
    #
    #     else:
    #         img_mix = img
    #         mask_mix_temp = mask_multi
    #         mask_seg_mix_temp = mask_seg_multi
    #
    #
    #     # if mix:
    #     #     img_mix = mixup*img_nc+(1-mixup)*img
    #     # else:
    #     #     img_mix = img
    #
    #     # fig, ax = plt.subplots(3,1, figsize=(20, 20))
    #     # fig.tight_layout()
    #     # # ax.axis('off')
    #     # # print(img[:,:,7].shape)
    #     # # sys.exit()
    #     #
    #     #
    #     # ax[0].imshow(img[:,:,7], cmap='gray')
    #     # ax[2].imshow(img_nc[:,:,7], cmap='gray')
    #     # # ax[2, 0].imshow(inner, cmap='gray')
    #     # # ax[2, 1].imshow(outer, cmap='gray')
    #     # # ax[2, 0].imshow(gt_seg_plq[0], cmap='gray')
    #     # # ax[2, 1].imshow(gt_seg_plq[1], cmap='gray')
    #     # # ax[2, 1].imshow(output_health[2], cmap='gray')
    #     # # ax[2, 2].imshow(output_unhealth[2], cmap='gray')
    #     # ax[1].imshow(img_mix[:,:,7], cmap='gray')
    #     # fig.tight_layout()
    #     # ax[0].set_title(path)
    #     # ax[2].set_title(path_nc)
    #     # ax[1].set_title(mixup)
    #     #
    #     # from os.path import exists
    #     # from os import mkdir
    #     # if not exists('mixup'):
    #     #     mkdir('mixup')
    #     # plt.savefig("mixup/mixup_{}.png".format(index))
    #
    #
    #
    #
    #
    #
    #
    #
    #     img_f=img_mix.astype(np.float32)
    #     img_f = img_f.transpose(2, 0, 1)
    #     end1 = time.time()
    #
    #     start1 = time.time()
    #     height, width = img.shape[0], img.shape[1]
    #
    #     orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
    #         snake_kins_utils.augment(
    #             img, self.split,
    #             snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
    #             snake_config.mean, snake_config.std, instance_polys
    #         )
    #
    #     instance_polys = self.get_valid_polys(instance_polys)
    #
    #     extreme_points = self.get_extreme_points(instance_polys)
    #
    #     #
    #     # instance_polys_nc = self.get_valid_polys(instance_polys_nc)
    #     #
    #     # extreme_points_nc = self.get_extreme_points(instance_polys_nc)
    #
    #     output_h, output_w = inp_out_hw[2:]
    #
    #     ct_hm = np.zeros([2, output_h, output_w], dtype=np.float32)
    #     ct_hm_nc = np.zeros([2, output_h, output_w], dtype=np.float32)
    #
    #     end2= time.time()
    #     # print("index:",index,"time_poly_prepare:",end2-start1)
    #
    #     wh = []
    #     ct_cls = []
    #     ct_ind = []
    #     wh_nc = []
    #     ct_cls_nc = []
    #     ct_ind_nc = []
    #
    #     # init
    #     i_it_4pys = []
    #     c_it_4pys = []
    #     i_gt_4pys = []
    #     c_gt_4pys = []
    #     # i_it_4pys_nc = []
    #     # c_it_4pys_nc = []
    #     # i_gt_4pys_nc = []
    #     # c_gt_4pys_nc = []
    #
    #     # evolution
    #     i_it_pys = []
    #     c_it_pys = []
    #     i_gt_pys = []
    #     c_gt_pys = []
    #     # i_it_pys_nc = []
    #     # c_it_pys_nc = []
    #     # i_gt_pys_nc = []
    #     # c_gt_pys_nc = []
    #
    #     # print("anno_len:",len(anno))
    #     # assert len(anno)==2
    #     # assert len(instance_polys)==2
    #
    #     for i in range(len(annotations)+len(annotations_nc) if mix else len(annotations)):
    #         cls_id = cls_ids[i]
    #
    #         instance_poly = instance_polys[i]
    #         instance_points = extreme_points[i]
    #
    #         for j in range(len(instance_poly)):
    #             poly = instance_poly[j]
    #             # assert not len(poly)
    #             extreme_point = instance_points[j]
    #
    #             x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
    #             x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
    #             bbox = [x_min, y_min, x_max, y_max]
    #             h, w = y_max - y_min + 1, x_max - x_min + 1
    #             if h <= 1 or w <= 1:
    #                 continue
    #
    #             decode_box = self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
    #             self.prepare_init(decode_box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
    #             self.prepare_evolution(poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)
    #     # print("len+len_nc:",len(annotations)+len(annotations_nc))
    #     # print("i_it_4pys_len:",len(i_it_4pys))
    #     # print("i_it_4pys[0]:",i_it_4pys[0].shape)
    #     # print("ct_hm:",ct_hm)
    #     # sys.exit()
    #
    #     # for i in range(len(annotations_nc)):
    #     #     cls_id_nc = cls_ids_nc[i]
    #     #
    #     #     instance_poly = instance_polys_nc[i]
    #     #     instance_points = extreme_points_nc[i]
    #     #
    #     #     for j in range(len(instance_poly)):
    #     #         poly = instance_poly[j]
    #     #         # assert not len(poly)
    #     #         extreme_point = instance_points[j]
    #     #
    #     #         x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
    #     #         x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
    #     #         bbox = [x_min, y_min, x_max, y_max]
    #     #         h, w = y_max - y_min + 1, x_max - x_min + 1
    #     #         if h <= 1 or w <= 1:
    #     #             continue
    #     #
    #     #         decode_box = self.prepare_detection(bbox, poly, ct_hm_nc, cls_id_nc, wh_nc, ct_cls_nc, ct_ind_nc)
    #     #         self.prepare_init(decode_box, extreme_point, i_it_4pys_nc, c_it_4pys_nc, i_gt_4pys_nc, c_gt_4pys_nc, output_h, output_w)
    #     #         self.prepare_evolution(poly, extreme_point, i_it_pys_nc, c_it_pys_nc, i_gt_pys_nc, c_gt_pys_nc)
    #     # print('i_gt_py_len_before', len(i_gt_pys))
    #     # assert len(i_gt_pys)==2
    #     start2=time.time()
    #     mask_3d = mask
    #
    #
    #     # mask_mix = np.zeros([5,96,96])
    #     # if mix:
    #     #     mask_mix[0][mask==1]=float(1-mixup)
    #     #     mask_mix[1][mask==2]=float(1-mixup)
    #     #     mask_mix[2][mask_nc==1]=float(mixup)
    #     #     mask_mix[3][mask_nc==2]=float(mixup)
    #     # else:
    #     #     if healthy:
    #     #         mask_mix[0][mask==1]=float(1)
    #     #         mask_mix[1][mask==2]=float(1)
    #     #     else:
    #     #         mask_mix[2][mask==1]=float(1)
    #     #         mask_mix[3][mask==2]=float(1)
    #     if self.split =="test":
    #         mask_mix = mask_nc_multi
    #
    #
    #
    #     # from os.path import exists
    #     # from os import mkdir
    #     # if not exists('mixup'):
    #     #     mkdir('mixup')
    #     # np.savetxt("mixup/mixup_0_{}.txt".format(index),mask_mix[0],fmt='%7.2f')
    #     # np.savetxt("mixup/mixup_1_{}.txt".format(index),mask_mix[1],fmt='%7.2f')
    #     # np.savetxt("mixup/mixup_2_{}.txt".format(index),mask_mix[2],fmt='%7.2f')
    #     # np.savetxt("mixup/mixup_3_{}.txt".format(index),mask_mix[3],fmt='%7.2f')
    #     #
    #     # sys.exit()
    #
    #
    #
    #
    #     mask_seg = np.ones(mask.shape)
    #     mask_1 = np.ones(mask.shape)
    #     # 1 for inner mask_1 is segmentation
    #     mask_back = np.ones(mask.shape)
    #     mask_wall = np.ones(mask.shape)
    #     mask_2 = np.ones(mask.shape)
    #
    #     mask_dis = np.zeros([3,96,96])
    #
    #     # 2 for outer mask_2 is segmentation
    #
    #     mask_1[mask == 1] = 0
    #     mask_2[mask == 2] = 0
    #
    #     mask_back[mask_2 == 1] = 0
    #     mask_wall = mask_2 - mask_1
    #
    #     mask_seg[mask_back == 1] = 0
    #     # background
    #     mask_seg[mask_wall == 1] = 1
    #     #  artery wall
    #     mask_seg[mask_1 == 1] = 2
    #
    #
    #     # sys.exit()
    #     for i in range(len(i_it_pys)):
    #         i_it_pys[i]=i_it_pys[i]/snake_config.down_ratio
    #     for i in range(len(c_it_pys)):
    #         c_it_pys[i]=c_it_pys[i]/snake_config.down_ratio
    #     for i in range(len(i_gt_pys)):
    #         i_gt_pys[i]=i_gt_pys[i]/snake_config.down_ratio
    #     for i in range(len(c_gt_pys)):
    #         c_gt_pys[i]=c_gt_pys[i]/snake_config.down_ratio
    #
    #     # for i in range(len(i_it_pys_nc)):
    #     #     i_it_pys_nc[i]=i_it_pys_nc[i]/snake_config.down_ratio
    #     # for i in range(len(c_it_pys_nc)):
    #     #     c_it_pys_nc[i]=c_it_pys_nc[i]/snake_config.down_ratio
    #     # for i in range(len(i_gt_pys_nc)):
    #     #     i_gt_pys_nc[i]=i_gt_pys_nc[i]/snake_config.down_ratio
    #     # for i in range(len(c_gt_pys_nc)):
    #     #     c_gt_pys_nc[i]=c_gt_pys_nc[i]/snake_config.down_ratio
    #     thickness_set=np.zeros(snake_config.poly_num)
    #
    #     ct_health = np.unique(mask_seg_plaque)
    #     label_health = np.array([1,0,0],dtype=float)
    #     if 3 in ct_health :
    #         label_health = np.array([0,1,0],dtype=float)
    #     if 4 in ct_health:
    #         label_health = np.array([0, 0, 1], dtype=float)
    #
    #     # ct_health = np.unique(mask_seg_plaque)
    #     # label_health = np.array([1,0,0,0],dtype=float)
    #     # if 3 in ct_health :
    #     #     label_health = np.array([0,1,0,0],dtype=float)
    #     # if 4 in ct_health:
    #     #     label_health = np.array([0, 0, 1,0], dtype=float)
    #     # if 3 in ct_health and 4 in ct_health:
    #     #     label_health = np.array([0, 0,0, 1], dtype=float)
    #     # label_health = np.zeros([15,4],dtype=float)
    #     # for i in range(len(label_health)):
    #     #     ct_health = np.unique(mask_seg_plaque[i])
    #     #     # print("i:",i)
    #     #     # print(" ct_health:",  ct_health)
    #     #     label_health[i] = np.array([1, 0, 0, 0], dtype=float)
    #     #     if 3 in ct_health:
    #     #         label_health[i] = np.array([0, 1, 0, 0], dtype=float)
    #     #     if 4 in ct_health:
    #     #         label_health[i] = np.array([0, 0, 1, 0], dtype=float)
    #     #     if 3 in ct_health and 4 in ct_health:
    #     #         label_health[i] = np.array([0, 0, 0, 1], dtype=float)
    #     # print("label_health:",label_health)
    #     # sys.exit()
    #
    #
    #
    #
    #     # ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg, 'path': path,
    #     #        'mask_seg_plaque': mask_seg_plaque}
    #     # ret = {'inp': img_f,'poly':instance_polys,'mask':mask,'mask_seg':mask_seg,'path':path,'risk_label':risk_onehot,'thickness_set':thickness_set}
    #     # ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg, 'path': path, 'mask_dis':sample_mask_dis,'mask_seg_plaque': mask_seg_plaque}
    #     ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask_mix_temp, 'mask_seg': mask_seg_plaque[7], 'path': path,
    #            'mask_dis': sample_mask_dis, 'mask_seg_plaque': mask_seg_mix_temp,'label_health':label_health}
    #     # print("inp_shape:",img_f.shape)
    #     # print("poly_shape:", len(instance_polys))
    #     # print("mask_shape:", mask.shape)
    #     # print("mask_seg_shape:", mask_seg.shape)
    #     # ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg, 'path': path}
    #     # print(path)
    #     # sys.exit()
    #     # print("inp_shape:",inp.shape)
    #     # print("instance_polys_shape:", len(instance_polys))
    #     # print("mask_shape:", mask.shape)
    #     # sys.exit()
    #     # print("len_poly:",len(ret['poly']))
    #
    #     detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
    #     init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
    #     evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
    #     # print("len_ig_t_pys:", len(i_gt_pys))
    #     # assert len(i_gt_pys)==2
    #     # print(i_gt_pys[1])
    #     t=0
    #     # for i in i_gt_pys:
    #     #     with open('py/py_%d.txt'%t, 'w') as outfile:
    #     #         np.savetxt(outfile, i, fmt='%s')
    #     #     t+=1
    #         # I'm writing a header here just for the sake of readability
    #         # Any line starting with "#" will be ignored by numpy.loadtxt
    #
    #
    #
    #
    #
    #     ret.update(detection)
    #     ret.update(init)
    #     ret.update(evolution)
    #     # visualize_utils.visualize_snake_detection(orig_img, ret)
    #     # visualize_utils.visualize_snake_evolution(orig_img, ret)
    #
    #     ct_num = len(ct_ind)
    #     meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'ct_num': ct_num}
    #     ret.update({'meta': meta})
    #
    #     # print(ret['poly'])
    #     end = time.time()
    #     # print("index:",index,"time_segmask:",end-start2)
    #     # print("index:",index,"time_all:", end - start)
    #     # print("----------------------------------------------")
    #     return ret












    # def __getitem__(self, index):
    #     # print("index:",index)
    #     # print("new")
    #     # print("index:",index)
    #     start = time.time()
    #     ann = self.anns[index]
    #     # print("annout:",ann)
    #
    #     anno, path, img_id = self.process_info(ann)
    #     # print("in _ batch")
    #     img, instance_polys, cls_ids, mask, instance_polys_1, annotations, sample_mask_dis,mask_seg_plaque,_= self.read_original_data(
    #         anno, path)
    #     # img, instance_polys, cls_ids, mask, instance_polys_1, annotations,sample_mask_dis = self.read_original_data(
    #     #     anno, path)
    #     # img, instance_polys, cls_ids ,mask,instance_polys_1,annotations,mask_seg_plaque= self.read_original_data(anno, path)
    #     # img, instance_polys, cls_ids, mask, instance_polys_1, annotations = self.read_original_data(
    #     #     anno, path)
    #     # print(index)
    #     # if len(anno)!=len(instance_polys):
    #     #     print("len(anno):",len(anno))
    #     #     print("len(instance_polys):", len(instance_polys))
    #     #     fig, ax = plt.subplots(3, figsize=(20, 10))
    #     #     fig.tight_layout()
    #     #     ax[0].imshow(mask, cmap='gray')
    #     #     ax[1].imshow(mask, cmap='gray')
    #     #     ax[2].imshow(mask, cmap='gray')
    #     #     colors_0 = np.array([
    #     #         [255, 127, 14],
    #     #         [10, 127, 255]
    #     #     ]) / 255.
    #     #     np.random.shuffle(colors_0)
    #     #     colors_0 = cycle(colors_0)
    #     #     if len(instance_polys) == 0:
    #     #         print("length has zero")
    #     #     # print("len(ex):", len(ex))
    #     #     # print("inp.shape",inp.shape)
    #     #     for i in range(len(instance_polys)):
    #     #         color = next(colors_0).tolist()
    #     #         for j in range(len(instance_polys[i])):
    #     #             ex1 = instance_polys[i] * snake_config.down_ratio
    #     #             color = next(colors_0).tolist()
    #     #             # print("For 0:", ex[label_0][0])
    #     #             # poly = py[label_0]
    #     #             poly = ex1[j]
    #     #
    #     #
    #     #             poly = np.append(poly, [poly[0]], axis=0)
    #     #             # print("i-th poly:", ex[i])
    #     #             ax[1].plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)
    #     #     for i in range(len(instance_polys_1)):
    #     #         color = next(colors_0).tolist()
    #     #         for j in range(len(instance_polys_1[i])):
    #     #             ex1 = instance_polys_1[i] * snake_config.down_ratio
    #     #
    #     #             # print("For 0:", ex[label_0][0])
    #     #             # poly = py[label_0]
    #     #             poly = ex1[j]
    #     #
    #     #
    #     #             poly = np.append(poly, [poly[0]], axis=0)
    #     #             # print("i-th poly:", ex[i])
    #     #             ax[2].plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)
    #     #     plt.savefig("bug_mask_1_{}.png".format(index))
    #
    #
    #
    #
    #
    #     img_f=img.astype(np.float32)
    #     img_f = img_f.transpose(2, 0, 1)
    #     end1 = time.time()
    #     # print("index:",index,"time_read_data:",end1-start)
    #     # print("img_f_shape:",img_f.shape)
    #     # sys.exit()
    #     # check=0
    #     # for py in instance_polys:
    #     #     for cood in py:
    #     #         for co in cood:
    #     #             assert co[0]<96 and co[1]<96
    #     #
    #     # print("ok_clear")
    #     # print("mask_shape:",mask.shape)
    #     # print("img_shape:", img.shape)
    #
    #     # pl["orig_poly"]=instance_polys
    #
    #     # print("instance_polys:", instance_polys)
    #     # with open('testimg.txt', 'w') as outfile:
    #     #     # I'm writing a header here just for the sake of readability
    #     #     # Any line starting with "#" will be ignored by numpy.loadtxt
    #     #     outfile.write('# Array shape: {0}\n'.format(img.shape))
    #     #
    #     #     # Iterating through a ndimensional array produces slices along
    #     #     # the last axis. This is equivalent to data[i,:,:] in this case
    #     #     for data_slice in img:
    #     #         # The formatting string indicates that I'm writing out
    #     #         # the values in left-justified columns 7 characters in width
    #     #         # with 2 decimal places.
    #     #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
    #     #
    #     #         # Writing out a break to indicate different slices...
    #     #         outfile.write('# New slice\n')
    #     start1 = time.time()
    #     height, width = img.shape[0], img.shape[1]
    #     # print("instance_polys:", instance_polys)
    #     # print("height:",height, " width:",width)
    #     orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
    #         snake_kins_utils.augment(
    #             img, self.split,
    #             snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
    #             snake_config.mean, snake_config.std, instance_polys
    #         )
    #     # for py in instance_polys:
    #     #     for cood in py:
    #     #         for co in cood:
    #     #             assert co[0]<96 and co[1]<96
    #     # print("flipped:",flipped)
    #     #
    #     # print("after_aug_ok_clear")
    #     # print("mask_shape:",mask.shape)
    #     # print("img_shape:", img.shape)
    #     # print(" trans_output:", trans_output)
    #     # sys.exit()
    #     # print("input_shape:",inp.shape)
    #     # # input_shape: (15, 192, 192)
    #     # sys.exit()
    #     # with open('test.txt', 'w') as outfile:
    #     #     # I'm writing a header here just for the sake of readability
    #     #     # Any line starting with "#" will be ignored by numpy.loadtxt
    #     #     outfile.write('# Array shape: {0}\n'.format(inp.shape))
    #     #
    #     #     # Iterating through a ndimensional array produces slices along
    #     #     # the last axis. This is equivalent to data[i,:,:] in this case
    #     #     for data_slice in inp:
    #     #         # The formatting string indicates that I'm writing out
    #     #         # the values in left-justified columns 7 characters in width
    #     #         # with 2 decimal places.
    #     #         np.savetxt(outfile, data_slice, fmt='%-7.2f')
    #     #
    #     #         # Writing out a break to indicate different slices...
    #     #         outfile.write('# New slice\n')
    #
    #     # with open("orig_poly.txt", "wb") as fp:  # Pickling
    #     #     pickle.dump(instance_polys, fp)
    #     # print("instance_polys:", instance_polys)
    #     # instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
    #     # print("transform_original_data")
    #     # for py in instance_polys:
    #     #     for cood in py:
    #     #         for co in cood:
    #     #             assert co[0]<96 and co[1]<96
    #     #
    #     # print("after_transform_original_data_ok_clear")
    #     # with open("trans_poly.txt", "wb") as fp:  # Pickling
    #     #     pickle.dump(instance_polys, fp)
    #
    #     instance_polys = self.get_valid_polys(instance_polys)
    #     # print("len_valid:",len(instance_polys))
    #     # assert len(instance_polys)==2
    #     # print("instance_polys:", instance_polys)
    #     # pl["trans_poly"]=instance_polys
    #     # with open('data.txt', 'w') as outfile:
    #     #     json.dump(pl, outfile)
    #     # print("instance_polys_len:",len( instance_polys))
    #     #
    #     # sys.exit()
    #     extreme_points = self.get_extreme_points(instance_polys)
    #     # assert len(extreme_points) == 2
    #     # print("extreme_points:",extreme_points)
    #
    #     # detection
    #     output_h, output_w = inp_out_hw[2:]
    #     # print("output_h:",output_h,"output_w:",output_w)
    #     # # output_h=input_h//4
    #     # sys.exit()
    #     ct_hm = np.zeros([2, output_h, output_w], dtype=np.float32)
    #     # mask = np.zeros([ output_h*snake_config.down_ratio, output_w*snake_config.down_ratio], dtype=np.float32)
    #     # for i in range(len(instance_polys)):
    #     #     # print("len_i:",len(instance_polys[i]))
    #     #     # 1
    #     #
    #     #     for cood in instance_polys[i][0]:
    #     #         # print("intcood:",cood)
    #     #         mask[int(cood[0]*2)][int(cood[1]*2)]=1+i
    #     # cv2.imwrite('labels.png', mask * 255//2)
    #     end2= time.time()
    #     # print("index:",index,"time_poly_prepare:",end2-start1)
    #
    #     wh = []
    #     ct_cls = []
    #     ct_ind = []
    #
    #     # init
    #     i_it_4pys = []
    #     c_it_4pys = []
    #     i_gt_4pys = []
    #     c_gt_4pys = []
    #
    #     # evolution
    #     i_it_pys = []
    #     c_it_pys = []
    #     i_gt_pys = []
    #     c_gt_pys = []
    #
    #     # print("anno_len:",len(anno))
    #     # assert len(anno)==2
    #     # assert len(instance_polys)==2
    #
    #     for i in range(len(annotations)):
    #         cls_id = cls_ids[i]
    #         # if len(anno) != len(instance_polys):
    #         #     print("len(anno):", len(anno))
    #         #
    #         #     print("len(instance_polys:", len(instance_polys))
    #         #     fig, ax = plt.subplots(3, figsize=(20, 10))
    #         #     fig.tight_layout()
    #         #     ax[0].imshow(mask, cmap='gray')
    #         #     ax[1].imshow(mask, cmap='gray')
    #         #     ax[2].imshow(mask, cmap='gray')
    #         #     colors_0 = np.array([
    #         #         [255, 127, 14],
    #         #         [10, 127, 255]
    #         #     ]) / 255.
    #         #     np.random.shuffle(colors_0)
    #         #     colors_0 = cycle(colors_0)
    #         #     if len(instance_polys) == 0:
    #         #         print("length has zero")
    #         #     # print("len(ex):", len(ex))
    #         #     # print("inp.shape",inp.shape)
    #         #     for i in range(len(instance_polys)):
    #         #         color = next(colors_0).tolist()
    #         #         for j in range(len(instance_polys[i])):
    #         #             ex1 = instance_polys[i] * snake_config.down_ratio
    #         #             color = next(colors_0).tolist()
    #         #             # print("For 0:", ex[label_0][0])
    #         #             # poly = py[label_0]
    #         #             poly = ex1[j]
    #         #
    #         #             poly = np.append(poly, [poly[0]], axis=0)
    #         #             # print("i-th poly:", ex[i])
    #         #             ax[1].plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)
    #         #     for i in range(len(instance_polys_1)):
    #         #         color = next(colors_0).tolist()
    #         #         for j in range(len(instance_polys_1[i])):
    #         #             ex1 = instance_polys_1[i] * snake_config.down_ratio
    #         #
    #         #             # print("For 0:", ex[label_0][0])
    #         #             # poly = py[label_0]
    #         #             poly = ex1[j]
    #         #
    #         #             poly = np.append(poly, [poly[0]], axis=0)
    #         #             # print("i-th poly:", ex[i])
    #         #             ax[2].plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)
    #         #     plt.savefig("bug_mask_2_{}.png".format(index))
    #         # if len(anno)!=len(instance_polys):
    #         #     np.savetxt("bug_mask.txt",mask)
    #         #     cv2.imwrite("bug_mask.png",mask*255)
    #         instance_poly = instance_polys[i]
    #         instance_points = extreme_points[i]
    #
    #         for j in range(len(instance_poly)):
    #             poly = instance_poly[j]
    #             # assert not len(poly)
    #             extreme_point = instance_points[j]
    #
    #             x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
    #             x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
    #             bbox = [x_min, y_min, x_max, y_max]
    #             h, w = y_max - y_min + 1, x_max - x_min + 1
    #             if h <= 1 or w <= 1:
    #                 continue
    #
    #             decode_box = self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
    #             self.prepare_init(decode_box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
    #             self.prepare_evolution(poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)
    #     # print('i_gt_py_len_before', len(i_gt_pys))
    #     # assert len(i_gt_pys)==2
    #     start2=time.time()
    #     mask_3d = mask
    #     mask= mask[7]
    #     mask_seg_plaque = mask_seg_plaque[7]
    #
    #     mask_seg = np.ones(mask.shape)
    #     mask_1 = np.ones(mask.shape)
    #     # 1 for inner mask_1 is segmentation
    #     mask_back = np.ones(mask.shape)
    #     mask_wall = np.ones(mask.shape)
    #     mask_2 = np.ones(mask.shape)
    #
    #     mask_dis = np.zeros([3,96,96])
    #
    #     # 2 for outer mask_2 is segmentation
    #
    #     mask_1[mask == 1] = 0
    #     mask_2[mask == 2] = 0
    #
    #     mask_back[mask_2 == 1] = 0
    #     mask_wall = mask_2 - mask_1
    #
    #     mask_seg[mask_back == 1] = 0
    #     # background
    #     mask_seg[mask_wall == 1] = 1
    #     #  artery wall
    #     mask_seg[mask_1 == 1] = 2
    #     # dis_2d with coord
    #     from os.path import exists
    #     from os import mkdir
    #     from os.path import join
    #     import scipy
    #     from lib.config import cfg, args
    #     from lib.utils.snake.snake_gcn_utils import uniform_upsample
    #
    #
    #     # # dis_coord
    #     # boundir = '/data/ugui0/antonio-t/BOUND/mask_dis_coord'
    #     # # boundir = '/data/ugui0/antonio-t/BOUND/feature_risk_label_config3'
    #     # # boundir = 'config-3_patients-100'
    #     # if not exists(boundir):
    #     #     mkdir(boundir)
    #     # # boundir = '/data/ugui0/antonio-t/BOUND/mask_dis/{}'.format(cfg.test.state)
    #     # # boundir = '/data/ugui0/antonio-t/BOUND/feature_risk_label_config3/{}'.format(cfg.test.state)
    #     # # boundir = 'config-3_patients-100/{}'.format(cfg.test.state)
    #     # if not exists(boundir):
    #     #     mkdir(boundir)
    #     # boundir = join(boundir, path.split('/')[5])
    #     # if not exists(boundir):
    #     #     mkdir(boundir)
    #     # boundir = join(boundir, path.split('/')[6])
    #     # if not exists(boundir):
    #     #     mkdir(boundir)
    #     # slice = path.split('/')[9]
    #     # # mask = batch['mask'][0].cpu().numpy()
    #     # # print('mask_shape:',mask.shape)
    #     # # sys.exit()
    #     # # temp = output['py'][-1].cpu().numpy()
    #     # # if len(temp) == 2:
    #     # #     thick, _ = self.myocardial_thickness(temp[0], temp[1])
    #     # # if len(temp) == 1:
    #     # #     thick, _ = self.myocardial_thickness(temp[0], temp[0])
    #     #
    #     # mask_seg = np.ones(mask.shape)
    #     # mask_1 = np.ones(mask.shape)
    #     # # 1 for inner mask_1 is segmentation
    #     # mask_back = np.ones(mask.shape)
    #     # mask_wall = np.ones(mask.shape)
    #     # mask_2 = np.ones(mask.shape)
    #     #
    #     # mask_dis = np.zeros([3, 96, 96])
    #     # mask_in_coord = np.zeros([2, 96, 96])
    #     # mask_out_coord = np.zeros([2, 96, 96])
    #     #
    #     # # 2 for outer mask_2 is segmentation
    #     #
    #     # mask_1[mask == 1] = 0
    #     # mask_2[mask == 2] = 0
    #     # self.fill(mask_1, (0, 0), 0)
    #     # self.fill(mask_2, (0, 0), 0)
    #     # mask_back[mask_2 == 1] = 0
    #     # mask_wall = mask_2 - mask_1
    #     #
    #     # mask_seg[mask_back == 1] = 0
    #     # # background
    #     # mask_seg[mask_wall == 1] = 1
    #     # #  artery wall
    #     # mask_seg[mask_1 == 1] = 2
    #     #
    #     # ex_in = np.argwhere(mask == 1)
    #     # ex_out = np.argwhere(mask == 2)
    #     # for i in range(96):
    #     #     for j in range(96):
    #     #         min_dist_in = 10000
    #     #         idx_in=0
    #     #         for k in range(len(ex_in)):
    #     #             a = np.array(ex_in[k])
    #     #             b = np.array([i, j])
    #     #             dst = scipy.spatial.distance.euclidean(a, b)
    #     #             if dst<min_dist_in:
    #     #                 min_dist_in = dst
    #     #                 idx_in = k
    #     #
    #     #         if len(ex_in):
    #     #             # print("idx_in:",idx_in[0][0])
    #     #
    #     #             assert scipy.spatial.distance.euclidean(np.array(ex_in[idx_in]), np.array([i, j])) == min_dist_in
    #     #             # print("assertclear")
    #     #             mask_in_coord[0][i][j] = ex_in[int(idx_in)][0]
    #     #             mask_in_coord[1][i][j] = ex_in[int(idx_in)][1]
    #     #
    #     #         else:
    #     #             min_dist_in = scipy.spatial.distance.euclidean(np.array([47, 47]), np.array([i, j]))
    #     #             mask_in_coord[0][i][j] = 47
    #     #             mask_in_coord[1][i][j] = 47
    #     #         # print("min_dist_in:", min_dist_in)
    #     #
    #     #         if mask_seg[i][j] == 2:
    #     #             mask_dis[0][i][j] = -1 * min_dist_in
    #     #         else:
    #     #             mask_dis[0][i][j] = min_dist_in
    #     #
    #     #         min_dist_out = 10000
    #     #         idx_out=0
    #     #         for z in range(len(ex_out)):
    #     #             a = np.array(ex_out[z])
    #     #             b = np.array([i, j])
    #     #             dst = scipy.spatial.distance.euclidean(a, b)
    #     #             if dst<min_dist_out:
    #     #                 min_dist_out = dst
    #     #                 idx_out = z
    #     #         if len(ex_out):
    #     #             # print("idx_out:", idx_out[0][0])
    #     #
    #     #             assert scipy.spatial.distance.euclidean(np.array(ex_out[idx_out]),
    #     #                                                     np.array([i, j])) == min_dist_out
    #     #             # print("assert clear out")
    #     #             mask_out_coord[0][i][j] = ex_out[idx_out][0]
    #     #             mask_out_coord[1][i][j] = ex_out[idx_out][1]
    #     #         else:
    #     #             min_dist_out = scipy.spatial.distance.euclidean(np.array([47, 47]), b)
    #     #             mask_out_coord[0][i][j] = 47
    #     #             mask_out_coord[1][i][j] = 47
    #     #
    #     #         if mask_seg[i][j] == 0:
    #     #             mask_dis[1][i][j] = min_dist_out
    #     #         else:
    #     #             mask_dis[1][i][j] = -1 * min_dist_out
    #     #
    #     # # print("temp_shape:",temp.shape)
    #     # # sys.exit()
    #     # # thick = batch['thickness_set'].cpu().numpy()
    #     # # risk_label = batch['risk_label'].cpu().numpy()
    #     # # # print('thick:',thick)
    #     # # print('thick_shape:', thick.shape)
    #     # # print('risk_label:', risk_label)
    #     # # print('risk_label_shape:', risk_label.shape)
    #     # # sys.exit()
    #     # # np.savetxt(join(boundir,slice)+'_feature.txt',thick)
    #     # # np.savetxt(join(boundir, slice) + '_label.txt', risk_label)
    #     # np.savetxt(join(boundir, slice) + '_inner.txt', mask_dis[0])
    #     # np.savetxt(join(boundir, slice) + '_outer.txt', mask_dis[1])
    #     # with open(join(boundir, slice) + '_inner_coord.txt', 'w') as outfile:
    #     #     for axis in mask_in_coord:
    #     #         np.savetxt(outfile, axis, fmt='%d')
    #     # with open(join(boundir, slice) + '_outer_coord.txt', 'w') as outfile:
    #     #     for axis in mask_out_coord:
    #     #         np.savetxt(outfile, axis, fmt='%d')
    #     # print("distance_finish",index)
    #     # # dis_coord
    #
    #
    #
    #
    #     from os.path import exists
    #     from os import mkdir
    #     from os.path import join
    #     import scipy
    #     from lib.config import cfg, args
    #     from lib.utils.snake.snake_gcn_utils import uniform_upsample
    #     # 3d distance
    #     # boundir = '/data/ugui0/antonio-t/BOUND/mask_dis_tiff_3D'
    #     # # boundir = '/data/ugui0/antonio-t/BOUND/feature_risk_label_config3'
    #     # # boundir = 'config-3_patients-100'
    #     # if not exists(boundir):
    #     #     mkdir(boundir)
    #     # # boundir = '/data/ugui0/antonio-t/BOUND/mask_dis/{}'.format(cfg.test.state)
    #     # # boundir = '/data/ugui0/antonio-t/BOUND/feature_risk_label_config3/{}'.format(cfg.test.state)
    #     # # boundir = 'config-3_patients-100/{}'.format(cfg.test.state)
    #     # if not exists(boundir):
    #     #     mkdir(boundir)
    #     # boundir = join(boundir,path.split('/')[5])
    #     # if not exists(boundir):
    #     #     mkdir(boundir)
    #     # boundir = join(boundir, path.split('/')[6])
    #     # if not exists(boundir):
    #     #     mkdir(boundir)
    #     # slice = path.split('/')[9]
    #     #
    #     # # print('mask_shape:',mask.shape)
    #     # # sys.exit()
    #     # # temp = output['py'][-1].cpu().numpy()
    #     # # if len(temp) == 2:
    #     # #     thick, _ = self.myocardial_thickness(temp[0], temp[1])
    #     # # if len(temp) == 1:
    #     # #     thick, _ = self.myocardial_thickness(temp[0], temp[0])
    #     #
    #     # mask_seg = np.ones(mask.shape)
    #     # mask_1 = np.ones(mask.shape)
    #     # # 1 for inner mask_1 is segmentation
    #     # mask_back = np.ones(mask.shape)
    #     # mask_wall = np.ones(mask.shape)
    #     # mask_2 = np.ones(mask.shape)
    #     #
    #     # mask_dis = np.zeros([3, 96, 96])
    #     #
    #     # # 2 for outer mask_2 is segmentation
    #     #
    #     # mask_1[mask == 1] = 0
    #     # mask_2[mask == 2] = 0
    #     # self.fill(mask_1, (0, 0), 0)
    #     # self.fill(mask_2, (0, 0), 0)
    #     # mask_back[mask_2 == 1] = 0
    #     # mask_wall = mask_2 - mask_1
    #     #
    #     # mask_seg[mask_back == 1] = 0
    #     # # background
    #     # mask_seg[mask_wall == 1] = 1
    #     # #  artery wall
    #     # mask_seg[mask_1 == 1] = 2
    #     #
    #     # ex_cal = np.argwhere(mask_3d == 3)
    #     # ex_noncal = np.argwhere(mask_3d == 4)
    #
    #     # id=np.unique(mask_3d)
    #     #
    #     # print("id:",id)
    #
    #     # if len(ex_cal):
    #     #     print(ex_cal)
    #     #     sys.exit()
    #     # else:
    #     #     print(index)
    #     #     return
    #
    #     #3d dis
    #     # label_cal=1
    #     # label_noncal =1
    #     # for i in range(96):
    #     #     for j in range(96):
    #     #         distance_cal = []
    #     #         for k in ex_cal:
    #     #             a = np.array([6*k[0],k[1],k[2]])
    #     #             b = np.array([6*(cfg.interval//2),i, j])
    #     #             dst = scipy.spatial.distance.euclidean(a, b)
    #     #             # print("a:", a)
    #     #             # print("b:", b)
    #     #             # print("dst:", dst)
    #     #             # sys.exit()
    #     #             distance_cal = np.append(distance_cal, dst)
    #     #         distance_cal = np.array(distance_cal)
    #     #         if len(distance_cal):
    #     #             min_dist_cal = np.min(distance_cal)
    #     #         else:
    #     #             min_dist_cal = 0.0
    #     #             label_cal = 0
    #     #         # print("min_dist_in:", min_dist_in)
    #     #
    #     #         mask_dis[0][i][j] = min_dist_cal
    #     #
    #     #         distance_noncal = []
    #     #         for z in ex_noncal:
    #     #             a = np.array([6*z[0],z[1],z[2]])
    #     #             b = np.array([6*(cfg.interval//2),i, j])
    #     #
    #     #             dst = scipy.spatial.distance.euclidean(a, b)
    #     #             # print("a:",a)
    #     #             # print("b:", b)
    #     #             # print("dst:", dst)
    #     #             # sys.exit()
    #     #             distance_noncal = np.append(distance_noncal, dst)
    #     #         distance_noncal = np.array(distance_noncal)
    #     #         if len(distance_noncal):
    #     #             min_dist_noncal = np.min(distance_noncal)
    #     #         else:
    #     #             min_dist_noncal = 0.0
    #     #             label_noncal=0
    #     #
    #     #         mask_dis[1][i][j] =  min_dist_noncal
    #
    #
    #     # mean_inner = mask_dis[0]-sample_mask_dis[0]
    #     # mean_outer = mask_dis[1]-sample_mask_dis[1]
    #     # mean_in = np.mean(mask_dis[0]-sample_mask_dis[0])
    #     # mean_out =np.mean(mask_dis[1]-sample_mask_dis[1])
    #     # print(index)
    #     # print("mean_in_00:",mask_dis[0][0][0],"and",sample_mask_dis[0][0][0])
    #     # print("mean_out_00:", mask_dis[1][0][0],"and",sample_mask_dis[1][0][0])
    #     # print("mean_in:",mean_in)
    #     # print("mean_out:", mean_out)
    #
    #     # print("temp_shape:",temp.shape)
    #     # sys.exit()
    #     # thick = batch['thickness_set'].cpu().numpy()
    #     # risk_label = batch['risk_label'].cpu().numpy()
    #     # # print('thick:',thick)
    #     # print('thick_shape:', thick.shape)
    #     # print('risk_label:', risk_label)
    #     # print('risk_label_shape:', risk_label.shape)
    #     # sys.exit()
    #     # np.savetxt(join(boundir,slice)+'_feature.txt',thick)
    #     # np.savetxt(join(boundir, slice) + '_label.txt', risk_label)
    #     # boundir_dis='test_dis'
    #     # np.savetxt(join(boundir_dis, slice) + '_inner_ave.txt', mask_dis[0]-sample_mask_dis[0])
    #     # np.savetxt(join(boundir_dis, slice) + '_outer_ave.txt', mask_dis[1]-sample_mask_dis[1])
    #
    #
    #     # 3d distance
    #     # slice = "%03d" % (int(path.split('/')[9]) + cfg.interval // 2)
    #     # np.savetxt(join(boundir, slice) + '_cal.txt', mask_dis[0])
    #     # np.savetxt(join(boundir, slice) + '_noncal.txt', mask_dis[1])
    #     # # cv2.imwrite(join(boundir, slice) + '_cal.tiff', mask_dis[0])
    #     # # cv2.imwrite(join(boundir, slice) + '_noncal.tiff', mask_dis[1])
    #     # if label_cal:
    #     #     print(join(boundir, slice))
    #     # if label_noncal:
    #     #     print(join(boundir, slice))
    #     # label_cal=[label_cal]
    #     #
    #     # label_noncal=[label_noncal]
    #     # np.savetxt(join(boundir, slice) + '_cal_exist.txt', label_cal)
    #     # np.savetxt(join(boundir, slice) + '_noncal_exist.txt', label_noncal)
    #
    #
    #
    #     # np.savetxt(join(boundir_dis, slice) + '_inner_new.txt', sample_mask_dis[0])
    #     # np.savetxt(join(boundir_dis, slice) + '_outer_new.txt', sample_mask_dis[1])
    #     # print(index)
    #
    #     # ex_in = np.argwhere(mask == 1)
    #     # ex_out = np.argwhere(mask == 2)
    #     # for i in range(96):
    #     #     for j in range(96):
    #     #         distance_in = []
    #     #         for k in ex_in:
    #     #             a = np.array(k)
    #     #             b = np.array([i,j])
    #     #             dst = scipy.spatial.distance.euclidean(a, b)
    #     #             distance_in = np.append(distance_in, dst)
    #     #         distance_in = np.array(distance_in)
    #     #         if len(distance_in):
    #     #             min_dist_in = np.min(distance_in)
    #     #         else:
    #     #             min_dist_in = scipy.spatial.distance.euclidean(np.array([47, 47]), np.array([i, j]))
    #     #         # print("min_dist_in:", min_dist_in)
    #     #
    #     #         if mask_seg[i][j] == 2:
    #     #             mask_dis[0][i][j] = -1*min_dist_in
    #     #         else:
    #     #             mask_dis[0][i][j] = min_dist_in
    #     #
    #     #
    #     #         distance_out = []
    #     #         for z in ex_out:
    #     #             a = np.array(z)
    #     #             b = np.array([i, j])
    #     #             dst = scipy.spatial.distance.euclidean(a, b)
    #     #             distance_out = np.append(distance_out, dst)
    #     #         distance_out = np.array(distance_out)
    #     #         if len(distance_out):
    #     #             min_dist_out = np.min(distance_out)
    #     #         else:
    #     #             min_dist_out = scipy.spatial.distance.euclidean(np.array([47,47]), b)
    #     #
    #     #         if mask_seg[i][j] == 0:
    #     #             mask_dis[1][i][j] = min_dist_out
    #     #         else:
    #     #             mask_dis[1][i][j] = -1*min_dist_out
    #     # mask_dis[1][i][j] = min_dist_out
    #
    #
    #
    #     # print(index)
    #     # with open('testmask_in_{}.txt'.format(index), 'w') as outfile:
    #     #     # I'm writing a header here just for the sake of readability
    #     #     # Any line starting with "#" will be ignored by numpy.loadtxt
    #     #     np.savetxt(outfile, mask_dis[0], fmt='%d')
    #     # with open('testmask_out_{}.txt'.format(index), 'w') as outfile:
    #     #     # I'm writing a header here just for the sake of readability
    #     #     # Any line starting with "#" will be ignored by numpy.loadtxt
    #     #     np.savetxt(outfile, mask_dis[1], fmt='%d')
    #     # sys.exit()
    #
    #
    #
    #
    #
    #     # total_distance_in_slice = []
    #     # for z in range(len(ords_in)):
    #     #     distance = []
    #     #     for k in range(len(ords_ex)):
    #     #         a = ords_in[z]
    #     #         a = np.array(a)
    #     #         # print a
    #     #         b = ords_ex[k]
    #     #         b = np.array(b)
    #     #         # dst = np.linalg.norm(a-b)
    #     #         dst = scipy.spatial.distance.euclidean(a, b)
    #     #         # pdb.set_trace()
    #     #         # if dst == 0:
    #     #         #     pdb.set_trace()
    #     #         distance = np.append(distance, dst)
    #     #     distance = np.array(distance)
    #     #     min_dist = np.min(distance)
    #     #     total_distance_in_slice = np.append(total_distance_in_slice, min_dist)
    #     #     total_distance_in_slice = np.array(total_distance_in_slice)
    #
    #
    #
    #     # lumen
    #     # mask_seg[mask == 1] = 0
    #     # # inner bound
    #     # mask_seg[mask == 2] = 0
    #     # outer bound
    #     # print("mask_seg:",mask_seg.shape)
    #     # with open('testmask_seg.txt', 'w') as outfile:
    #     #     # I'm writing a header here just for the sake of readability
    #     #     # Any line starting with "#" will be ignored by numpy.loadtxt
    #     #     np.savetxt(outfile, mask_seg, fmt='%d')
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #     # mask_f=np.zeros([ output_h*snake_config.down_ratio, output_w*snake_config.down_ratio], dtype=np.float32)
    #     # print("mask_f_shape:",mask_f.shape)
    #     # print("mask.shape:",mask.shape)
    #     # cv2.imwrite('labels2.png', mask * 125)
    #     # fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    #     # fig.tight_layout()
    #     # # ax.axis('off')
    #     #
    #     # ax[0, 1].imshow(inp[7], cmap='gray')
    #     #
    #     # ax[1, 1].imshow(mask, cmap='gray')
    #     # # ax[0,0].hist(gt_thickness_set, bins=50)
    #     # # ax[1,0].hist(
    #     # #     thickness_set, bins=50)
    #     #
    #     # colors_0 = np.array([
    #     #     [255, 127, 14]
    #     # ]) / 255.
    #     # np.random.shuffle(colors_0)
    #     # colors_0 = cycle(colors_0)
    #     #
    #     # color = next(colors_0).tolist()
    #     #
    #     # # poly = py[label_0]
    #     # poly = i_gt_pys[0]/snake_config.down_ratio
    #     #
    #     # poly = np.append(poly, [poly[0]], axis=0)
    #     #
    #     # in_poly = i_it_pys[0] / snake_config.down_ratio
    #     # in_poly = np.append(in_poly, [in_poly[0]], axis=0)
    #     # # print("i-th poly:", ex[i])
    #     # ax[1, 1].plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)
    #     # ax[1, 1].plot(in_poly[:, 0], in_poly[:, 1], color=color, linewidth=2)
    #     #
    #     # # for i in range(len(ex)):
    #     # #     if label[i]:
    #     # #         continue
    #     # #     color = next(colors_0).tolist()
    #     # #     print("For 0:", ex[i][0])
    #     # #     poly = ex[i]
    #     # #     poly = np.append(poly, [poly[0]], axis=0)
    #     # #     # print("i-th poly:", ex[i])
    #     # #     ax[0].plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)
    #     # #     break
    #     # colors_1 = np.array([
    #     #
    #     #     [140, 86, 75]
    #     # ]) / 255.
    #     # np.random.shuffle(colors_1)
    #     # colors_1 = cycle(colors_1)
    #     # color = next(colors_1).tolist()
    #     # # poly = py[label_1]
    #     # poly = i_gt_pys[1]/snake_config.down_ratio
    #     # in_poly = i_it_pys[1] / snake_config.down_ratio
    #     # poly = np.append(poly, [poly[0]], axis=0)
    #     # in_poly = np.append(in_poly, [in_poly[0]], axis=0)
    #     # # print("i-th poly:", ex[i])
    #     # ax[1, 1].plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)
    #     # ax[1, 1].plot(in_poly[:, 0], in_poly[:, 1], color=color, linewidth=2)
    #     # # ax[1,1].set_title('ct_score_0: {:.3f} '.format(float(max_score_0))+'ct_score_1: {:.3f} '.format(float(max_score_1))+'hdf_0: {:.3f} '.format(float(hdf[0]))+'hdf_1: {:.3f} '.format(float(hdf[1]))+'gt_ave_thickness: {:.3f} '.format(gt_ave_thickness))
    #     #
    #     # # for i in range(len(ex)):
    #     # #     if not label[i]:
    #     # #         continue
    #     # #     color = next(colors_1).tolist()
    #     # #     print("For 1:",ex[i][0])
    #     # #     poly = ex[i]
    #     # #     poly = np.append(poly, [poly[0]], axis=0)
    #     # #     # print("i-th poly:", ex[i])
    #     # #     ax[0].plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)
    #     # #     break
    #     #
    #     # # x_min, y_min, x_max, y_max = box[i]
    #     # # ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color='w', linewidth=0.2)
    #     # #
    #     # # colors_gt = np.array([
    #     # #     [31, 119, 180],
    #     # #     [0, 0, 0]
    #     # # ]) / 255.
    #     # # print("len(gt):", len(gt))
    #     # # ax[1].plot(gt_0[:][0], gt_0[:][1], color=color1, linewidth=2)
    #     # # ax[1].plot(gt_1[:][0], gt_1[:][1], color=color1, linewidth=2)
    #     # # for i in range(len(gt_py)):
    #     # #     color1 = colors_gt[0].tolist()
    #     # #     if ct_cls[i]:
    #     # #         color1 = colors_gt[1].tolist()
    #     # #
    #     # #     # poly1 = gt_py[i]
    #     # #     poly1 = gt[i]
    #     # #     poly1 = np.append(poly1, [poly1[0]], axis=0)
    #     # #     # print("i-th poly:", gt[i])
    #     # #     ax[0, 1].plot(poly1[:, 0], poly1[:, 1], color=color1, linewidth=2)
    #     # # for i in range(len(gt_1)):
    #     # #     color1 = next(colors_gt).tolist()
    #     # #     poly1 = gt_1[i]
    #     # #     poly1 = np.append(poly1, [poly1[0]], axis=0)
    #     # #     # print("i-th poly:", gt[i])
    #     # #     ax[1].plot(poly1[:, 0], poly1[:, 1], color=color1, linewidth=2)
    #     #
    #     # plt.savefig("demo_batch/demo.png" )
    #     # sys.exit()
    #     # with open('testimg.txt', 'w') as outfile:
    #     #     # I'm writing a header here just for the sake of readability
    #     #     # Any line starting with "#" will be ignored by numpy.loadtxt
    #     #     np.savetxt(outfile, mask, fmt='%d')
    #
    #     # sys.exit()
    #     for i in range(len(i_it_pys)):
    #         i_it_pys[i]=i_it_pys[i]/snake_config.down_ratio
    #     for i in range(len(c_it_pys)):
    #         c_it_pys[i]=c_it_pys[i]/snake_config.down_ratio
    #     for i in range(len(i_gt_pys)):
    #         i_gt_pys[i]=i_gt_pys[i]/snake_config.down_ratio
    #     for i in range(len(c_gt_pys)):
    #         c_gt_pys[i]=c_gt_pys[i]/snake_config.down_ratio
    #     thickness_set=np.zeros(snake_config.poly_num)
    #     # if len(i_gt_pys)==2:
    #     #     thickness_set, _ = self.myocardial_thickness(i_gt_pys[0],i_gt_pys[1])
    #
    #     # print("thickness_set_len:",len(thickness_set))
    #     # print("thickness_set:", thickness_set)
    #     # print("risk_label:", risk_label)
    #     # sys.exit()
    #     # risk_onehot = np.zeros(4)
    #     # risk_onehot[int(risk_label)]=1
    #     # print("risk_onehot:", risk_onehot)
    #     # np.savetxt(os.path.join("test_seg_skip", "%03d" % (index)) + '.txt', mask_seg_plaque,fmt="%d")
    #     ct_health = np.unique(mask_seg_plaque)
    #     label_health = np.array([1,0,0],dtype=float)
    #     if 3 in ct_health :
    #         label_health = np.array([0,1,0],dtype=float)
    #     if 4 in ct_health:
    #         label_health = np.array([0, 0, 1], dtype=float)
    #
    #     # ct_health = np.unique(mask_seg_plaque)
    #     # label_health = np.array([1,0,0,0],dtype=float)
    #     # if 3 in ct_health :
    #     #     label_health = np.array([0,1,0,0],dtype=float)
    #     # if 4 in ct_health:
    #     #     label_health = np.array([0, 0, 1,0], dtype=float)
    #     # if 3 in ct_health and 4 in ct_health:
    #     #     label_health = np.array([0, 0,0, 1], dtype=float)
    #     # label_health = np.zeros([15,4],dtype=float)
    #     # for i in range(len(label_health)):
    #     #     ct_health = np.unique(mask_seg_plaque[i])
    #     #     # print("i:",i)
    #     #     # print(" ct_health:",  ct_health)
    #     #     label_health[i] = np.array([1, 0, 0, 0], dtype=float)
    #     #     if 3 in ct_health:
    #     #         label_health[i] = np.array([0, 1, 0, 0], dtype=float)
    #     #     if 4 in ct_health:
    #     #         label_health[i] = np.array([0, 0, 1, 0], dtype=float)
    #     #     if 3 in ct_health and 4 in ct_health:
    #     #         label_health[i] = np.array([0, 0, 0, 1], dtype=float)
    #     # print("label_health:",label_health)
    #     # sys.exit()
    #
    #
    #
    #
    #     # ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg, 'path': path,
    #     #        'mask_seg_plaque': mask_seg_plaque}
    #     # ret = {'inp': img_f,'poly':instance_polys,'mask':mask,'mask_seg':mask_seg,'path':path,'risk_label':risk_onehot,'thickness_set':thickness_set}
    #     # ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg, 'path': path, 'mask_dis':sample_mask_dis,'mask_seg_plaque': mask_seg_plaque}
    #     ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg_plaque, 'path': path,
    #            'mask_dis': sample_mask_dis, 'mask_seg_plaque': mask_seg_plaque,'label_health':label_health}
    #     # print("inp_shape:",img_f.shape)
    #     # print("poly_shape:", len(instance_polys))
    #     # print("mask_shape:", mask.shape)
    #     # print("mask_seg_shape:", mask_seg.shape)
    #     # ret = {'inp': img_f, 'poly': instance_polys, 'mask': mask, 'mask_seg': mask_seg, 'path': path}
    #     # print(path)
    #     # sys.exit()
    #     # print("inp_shape:",inp.shape)
    #     # print("instance_polys_shape:", len(instance_polys))
    #     # print("mask_shape:", mask.shape)
    #     # sys.exit()
    #     # print("len_poly:",len(ret['poly']))
    #
    #     detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
    #     init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
    #     evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
    #     # print("len_ig_t_pys:", len(i_gt_pys))
    #     # assert len(i_gt_pys)==2
    #     # print(i_gt_pys[1])
    #     t=0
    #     # for i in i_gt_pys:
    #     #     with open('py/py_%d.txt'%t, 'w') as outfile:
    #     #         np.savetxt(outfile, i, fmt='%s')
    #     #     t+=1
    #     # I'm writing a header here just for the sake of readability
    #     # Any line starting with "#" will be ignored by numpy.loadtxt
    #
    #
    #
    #
    #
    #     ret.update(detection)
    #     ret.update(init)
    #     ret.update(evolution)
    #     # visualize_utils.visualize_snake_detection(orig_img, ret)
    #     # visualize_utils.visualize_snake_evolution(orig_img, ret)
    #
    #     ct_num = len(ct_ind)
    #     meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'ct_num': ct_num}
    #     ret.update({'meta': meta})
    #
    #     # print(ret['poly'])
    #     end = time.time()
    #     # print("index:",index,"time_segmask:",end-start2)
    #     # print("index:",index,"time_all:", end - start)
    #     # print("----------------------------------------------")
    #     return ret

    def __len__(self):
        return len(self.anns)

