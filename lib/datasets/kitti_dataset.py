import os
import numpy as np
import torch.utils.data as torch_data
import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
from PIL import Image
from lib.config import cfg


class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split = 'train'):
        self.split = split
        is_test = self.split == 'test'
        self.imageset_dir = os.path.join(root_dir, 'KITTI', 'object', 'testing' if is_test else 'training')

        split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', split + '.txt')
        self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.num_sample = self.image_idx_list.__len__()

        self.image_dir = os.path.join(self.imageset_dir, 'image_2')
        self.lidar_dir = os.path.join(self.imageset_dir, 'velodyne')
        self.calib_dir = os.path.join(self.imageset_dir, 'calib')
        self.label_dir = os.path.join(self.imageset_dir, 'label_2')
        self.plane_dir = os.path.join(self.imageset_dir, 'planes')
        if cfg.USE_IM_DEPTH:
            self.depth_dir = os.path.join(self.imageset_dir, 'depth')
            self.pseudo_lidar_dir = os.path.join(self.imageset_dir, 'pseudo_lidar')

        if cfg.USE_PAINTING_SCORE:
            # self.painting_score_lidar_dir = os.path.join('/data2/zheliu/TPAMI_rebuttal_2022/img_output/pretrained_img_scores', cfg.CLASSES)
            self.painting_score_lidar_dir = os.path.join('/data3/kitti_mask/soft_mask_10e')

        if cfg.USE_PAINTING_FEAT:
            self.painting_feat_lidar_dir = os.path.join('/data2/zheliu/TPAMI_rebuttal_2022/img_output/pretrained_img_feats', cfg.CLASSES)

        self.mask_dir = os.path.join(self.imageset_dir, 'train_mask')

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # Don't need to permute while using grid_sample
        self.image_hw_with_padding_np = np.array([1280., 384.])

    def get_image(self, idx):
        assert False, 'DO NOT USE cv2 NOW, AVOID DEADLOCK'
        import cv2
        # cv2.setNumThreads(0)  # for solving deadlock when switching epoch
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    def get_image_rgb_with_normal(self, idx):
        """
        return img with normalization in rgb mode
        :param idx:
        :return: imback(H,W,3)
        """
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file).convert('RGB')
        im = np.array(im).astype(np.float)
        im = im / 255.0
        im -= self.mean
        im /= self.std
        #print(im.shape)
        # ~[-2,2]
        # im = im[:, :, ::-1]
        # make same size padding with 0
        ##################################
        if cfg.USE_IM_DEPTH:
            imback = np.zeros([384, 1280, 4], dtype=np.float)  ##  imback = np.zeros([384, 1280, 4], dtype=np.float)
            imback[:im.shape[0], :im.shape[1], 0:3] = im

            depth_file = os.path.join(self.depth_dir, '%06d.png' % idx)
            depth = np.array(Image.open(depth_file)).astype(np.float32)
            depth = depth / 256.0   ### patchnet里面处理的,  可以得到真实的depth
            imback[:im.shape[0], :im.shape[1], 3] = depth/100.0  ## depth的范围[0,100], 也可能小于0或者大于100, 因此除以100试试

        else:
            imback = np.zeros([384, 1280, 3], dtype = np.float)
            imback[:im.shape[0], :im.shape[1], :] = im
        ##################################
        # imback = np.zeros([384, 1280, 3], dtype = np.float)
        # imback[:im.shape[0], :im.shape[1], :] = im

        return imback  # (H,W,3) RGB mode

    def get_image_shape_with_padding(self, idx = 0):
        return 384, 1280, 3

    def get_KINS_car_mask(self, idx):

        if cfg.CLASSES == 'Car':
            LivingThing = [1, 2, 3, 5, 6, 8]
            vehicles = [4, 7]
        elif cfg.CLASSES == 'Pedestrian':
            LivingThing = [1, 3, 4, 5, 6, 7, 8]
            vehicles = [2]
        elif cfg.CLASSES == 'Cyclist':
            LivingThing = [2, 3, 4, 5, 6, 7, 8]
            vehicles = [1]

        # LivingThing = [1, 2, 3, 5, 6, 8]
        # vehicles = [4, 7]
        '''
        [(1, {'supercategory': 'Living Thing', 'id': 1, 'name': 'cyclist'}),
         (2, {'supercategory': 'Living Thing', 'id': 2, 'name': 'pedestrian'}), 
         (4, {'supercategory': 'vehicles', 'id': 4, 'name': 'car'}), 
         (5, {'supercategory': 'vehicles', 'id': 5, 'name': 'tram'}), 
         (6, {'supercategory': 'vehicles', 'id': 6, 'name': 'truck'}), 
         (7, {'supercategory': 'vehicles', 'id': 7, 'name': 'van'}), 
         (8, {'supercategory': 'vehicles', 'id': 8, 'name': 'misc'})]
        '''
        cat_mask = np.load(os.path.join(self.mask_dir, '%06d.npy' % idx))
        ret = -np.ones([384, 1280], dtype=np.float32)
        for id in LivingThing:
            cat_mask[cat_mask == id] = 0.0

        for id in vehicles:
            cat_mask[cat_mask == id] = 1.0  # 255 #1

        ret[:cat_mask.shape[0], :cat_mask.shape[1]] = cat_mask
        return ret

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype = np.float32).reshape(-1, 4)

    def get_pseudo_lidar(self, idx):
        pseudo_lidar_file = os.path.join(self.pseudo_lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(pseudo_lidar_file)
        return np.fromfile(pseudo_lidar_file, dtype=np.float32).reshape(-1, 3)

    def get_painting_score_lidar(self, idx):
        painting_score_file = os.path.join(self.painting_score_lidar_dir, '%06d.npy' % idx) #'%04d.npy'
        assert os.path.exists(painting_score_file)
        return np.load(painting_score_file)

    def get_painting_feat_lidar(self, idx):
        painting_score_file = os.path.join(self.painting_feat_lidar_dir, '%0d.npy' % idx)
        assert os.path.exists(painting_score_file)
        return np.load(painting_score_file)


    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_utils.get_objects_from_label(label_file)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.plane_dir, '%06d.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
