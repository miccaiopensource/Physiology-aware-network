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

        padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_binary_mask, 0.5)
        contours = np.subtract(np.array(contours, dtype=object), 1)
        for contour in contours:
            contour = self.close_contour(contour)
            contour = measure.approximate_polygon(contour, tolerance)
            if len(contour) < 3:
                continue
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
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
        total_distance_in_slice = []
        for z in range(len(ords_in)):
            distance = []
            for k in range(len(ords_ex)):
                a = ords_in[z]
                a = np.array(a)
                b = ords_ex[k]
                b = np.array(b)
                dst = scipy.spatial.distance.euclidean(a, b)
                distance = np.append(distance, dst)
            distance = np.array(distance)
            min_dist = np.min(distance)
            total_distance_in_slice = np.append(total_distance_in_slice, min_dist)
            total_distance_in_slice = np.array(total_distance_in_slice)


        average_distance_in_slice = np.mean(total_distance_in_slice)
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

            else:
                mask3d_dis_noncal = mask3d_dis_noncal + 68
            mask3d_dis_cal = os.path.join(mask3d_dis_dir, slice_3d) + '_cal.txt'
            mask3d_dis_cal_exist = os.path.join(mask3d_dis_dir, slice_3d) + '_cal_exist.txt'
            mask3d_dis_cal = np.loadtxt(mask3d_dis_cal)
            mask3d_dis_cal_exist = np.loadtxt(mask3d_dis_cal_exist)
            if mask3d_dis_cal_exist:
                mask3d_dis_cal[mask3d_dis_noncal>68]=68
            else:
                mask3d_dis_cal = mask3d_dis_noncal + 68
            mask_dis_inner =os.path.join(mask_dis_dir , slice) + '_inner.tiff'
            mask_dis_outer = os.path.join(mask_dis_dir , slice) + '_outer.tiff'
            mask_dis = np.zeros([3,96,96])
            mask_dis[0]=np.loadtxt(mask_dis_inner)
            mask_dis[1] = np.loadtxt(mask_dis_outer)
            for i in image_path_axis.split('/')[:-3]:
                risk_label_path =risk_label_path+i+'/'
            for i in image_path_axis.split('/')[:-1]:
                image_path = image_path + i + '/'
            risk_label_path = risk_label_path+'risk_labels.txt'

            slice_files = sorted(
                [file for file in listdir(image_path) if file.endswith('.tiff') and not file.startswith('.')])


            start_file, end_file = slice_files[0], slice_files[-1]

            start_num= int(start_file.split('.tiff')[0])
            end_num = int(end_file.split('.tiff')[0])

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
        sample_mask_f = sample_mask
        sample_mask = sample_mask[cfg.interval//2]
        mask_seg_ids = np.unique(sample_mask_seg)
        if 4 not in mask_seg_ids:
            healthy = 1
        else:
            healthy = 0
        end2 = time.time()
        img = np.transpose(sample_img,(1,2,0))
        instance_ids = np.unique(sample_mask)

        polys = []
        annotations=[]
        for instance_id in instance_ids:
            if instance_id == 0 or instance_id == 3 or instance_id == 4:  # background or edge, pass
                continue

            temp = np.zeros(sample_mask.shape)

            temp[sample_mask == instance_id] = 1
            self.fill(temp, (0, 0), 2)
            temp_f = np.ones(sample_mask.shape)
            temp_f[temp == 2] = 0

            instance = temp_f

            poly = self.binary_mask_to_polygon(instance)
            if len(poly) == 0:
                continue
            annos = {'segmentation': poly,'category_id':int(instance_id)-1}
            annotations.append(annos)



        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in annotations
                          if not isinstance(obj['segmentation'], dict)]
        instance_polys_1 = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno
                          if not isinstance(obj['segmentation'], dict)]
        cls_ids = [obj['category_id'] for obj in annotations]
        end3 = time.time()

        return img, instance_polys, cls_ids,sample_mask_f,instance_polys_1,annotations,sample_mask_dis,sample_mask_seg,healthy

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

            polys = snake_kins_utils.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)

        for py in instance_polys:
            for cood in py:
                for co in cood:
                    assert co[0]<96 and co[1]<96
        return instance_polys_

    def get_valid_polys(self, instance_polys):
        instance_polys_ = []


        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            polys = snake_kins_utils.filter_tiny_polys(instance)
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

        start = time.time()
        ann = self.anns[index]



        anno, path, img_id = self.process_info(ann)

        img, instance_polys, cls_ids, mask, instance_polys_1, annotations, sample_mask_dis,mask_seg_plaque,healthy = self.read_original_data(
            anno, path)


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


        output_h, output_w = inp_out_hw[2:]

        ct_hm = np.zeros([2, output_h, output_w], dtype=np.float32)
        ct_hm_nc = np.zeros([2, output_h, output_w], dtype=np.float32)

        end2= time.time()

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
        i_it_pys = []
        c_it_pys = []
        i_gt_pys = []
        c_gt_pys = []

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
        start2=time.time()
        mask_3d = mask




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
        thickness_set=np.zeros(snake_config.poly_num)

        ct_health = np.unique(mask_seg_plaque)
        label_health = np.array([1,0,0],dtype=float)
        if 3 in ct_health :
            label_health = np.array([0,1,0],dtype=float)
        if 4 in ct_health:
            label_health = np.array([0, 0, 1], dtype=float)

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

        detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
        evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}





        ret.update(detection)
        ret.update(init)
        ret.update(evolution)

        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'ct_num': ct_num}
        ret.update({'meta': meta})


        return ret



    def __len__(self):
        return len(self.anns)

