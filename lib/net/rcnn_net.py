import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetSAModule
from lib.rpn.proposal_target_layer import ProposalTargetLayer
import pointnet2_lib.pointnet2.pytorch_utils as pt_utils
import lib.utils.loss_utils as loss_utils
from lib.config import cfg

import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from torch.nn.functional import grid_sample
from lib.utils.sample2grid import sample2grid_F,sample2GaussianGrid_F, sample2BilinearGrid_F


BatchNorm2d = nn.BatchNorm2d
def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes//2, 1)
        self.bn1 = BatchNorm2d(inplanes//2 )
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(inplanes//2, outplanes, stride)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        return out

def Feature_Gather(feature_map, xy):
    """
    :param xy:(B,N,2)  归一化到[-1,1]
    :param feature_map:(B,C,H,W)
    :return:
    """
    # use grid_sample for this.
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)

    interpolate_feature = grid_sample(feature_map, xy)  # (B,C,1,N)

    return interpolate_feature.squeeze(2) # (B,C,N)

class Fusion_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):

        super(Fusion_Conv, self).__init__()

        self.conv1 = torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        #print(point_features.shape, img_features.shape)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


#================addition attention (add)=======================#
class IA_Layer(nn.Module):
    def __init__(self, channels):
        print('##############ADDITION ATTENTION(ADD) RCNN#########')
        super(IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                    nn.BatchNorm1d(self.pc),
                                    nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)


    def forward(self, img_feas, point_feas):
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BCN->BNC->(BN)C'
        # print(img_feas)
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        att = F.sigmoid(self.fc3(F.tanh(ri + rp))) #BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1) #B1N
        # print(img_feas.size(), att.size())

        img_feas_new = self.conv1(img_feas)
        out = img_feas_new * att

        return out


class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes, num_points = None):
        super(Atten_Fusion_Conv, self).__init__()

        self.IA_Layer = IA_Layer(channels = [inplanes_I, inplanes_P])
        #self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)

        img_features = self.IA_Layer(img_features, point_features)
        #print("img_features:", img_features.shape)

        # fusion_features = img_features + point_features  ###   ori
        # fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))  ###   ori

        fusion_features = torch.cat([point_features, img_features], dim=1)  ###   new 7.12
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))  ###   new 7.12

        return fusion_features


class Fusion_Cross_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):

        super(Fusion_Cross_Conv, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.conv1 = conv3x3(inplanes, outplanes, stride=1) #torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = BatchNorm2d(outplanes)
        print('############## USE RCNN CROSS FUSION!!')
        print('############## USE RCNN CROSS FUSION!!')
        print('############## USE RCNN CROSS FUSION!!')
        #self.conv2 = conv3x3(outplanes, outplanes, stride=1)

    def forward(self, point_features, img_features):
        #print(point_features.shape, img_features.shape)
        fusion_features = torch.cat([point_features, img_features], dim=1)

        # print('##############fusion_features:', fusion_features.shape)
        # print('##############inplanes:', self.inplanes)
        # print('##############outplanes:', self.outplanes)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))
        #fusion_features = self.conv2(fusion_features)

        return fusion_features

def grid_sample_reverse(point_feature, xy, img_shape):

    # print('#######point_feature:', point_feature.shape)
    # print('#######xy:', xy.shape)
    # print('#######size:', size)
    size = [i for i in img_shape]
    size[1] = point_feature.shape[1]
    project_point2img = sample2BilinearGrid_F(point_feature, xy, size)

    return project_point2img


class RCNNNet(nn.Module):
    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        if cfg.RCNN.USE_RPN_FEATURES:
            self.rcnn_input_channel = 3 + int(cfg.RCNN.USE_INTENSITY) + int(cfg.RCNN.USE_MASK) + int(cfg.RCNN.USE_DEPTH)
            self.xyz_up_layer = pt_utils.SharedMLP([self.rcnn_input_channel] + cfg.RCNN.XYZ_UP_LAYER,
                                                   bn=cfg.RCNN.USE_BN)
            c_out = cfg.RCNN.XYZ_UP_LAYER[-1]
            self.merge_down_layer = pt_utils.SharedMLP([c_out * 2, c_out], bn=cfg.RCNN.USE_BN)

        for k in range(cfg.RCNN.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + cfg.RCNN.SA_CONFIG.MLPS[k]

            npoint = cfg.RCNN.SA_CONFIG.NPOINTS[k] if cfg.RCNN.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=npoint,
                    radius=cfg.RCNN.SA_CONFIG.RADIUS[k],
                    nsample=cfg.RCNN.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.RCNN.USE_BN
                )
            )
            channel_in = mlps[-1]

        if cfg.DEEP_RCNN_FUSION:
            self.Img_Block_RCNN = nn.ModuleList()
            IMG_CHANNEL = int(cfg.RCNN_IMG_CHANNEL // 2)
            self.Img_Block_RCNN.append(BasicBlock(32, IMG_CHANNEL, stride=2))
            self.Img_Block_RCNN.append(BasicBlock(IMG_CHANNEL, IMG_CHANNEL*2, stride=2))

            if cfg.LI_FUSION.ENABLED:
                self.Fusion_Conv_RCNN = nn.ModuleList()
                self.Fusion_Conv_RCNN.append(Atten_Fusion_Conv(IMG_CHANNEL,128,128))
                self.Fusion_Conv_RCNN.append(Atten_Fusion_Conv(IMG_CHANNEL*2,256,256))

            else:
                self.Fusion_Conv_RCNN = nn.ModuleList()
                self.Fusion_Conv_RCNN.append(Fusion_Conv(IMG_CHANNEL+128,128))
                self.Fusion_Conv_RCNN.append(Fusion_Conv(IMG_CHANNEL*2+256,256))

            if cfg.CROSS_FUSION:
                self.Cross_Fusion = nn.ModuleList()
                self.Cross_Fusion.append(Fusion_Cross_Conv(IMG_CHANNEL+128, IMG_CHANNEL))
                self.Cross_Fusion.append(Fusion_Cross_Conv(IMG_CHANNEL*2+256, IMG_CHANNEL*2))

        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0],
                                                                           gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            self.cls_loss_func = F.binary_cross_entropy
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':
            cls_weight = torch.from_numpy(cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduce=False, weight=cls_weight)
        else:
            raise NotImplementedError

        if cfg.USE_IOU_BRANCH:
            iou_branch = []
            iou_branch.append(pt_utils.Conv1d(channel_in, cfg.RCNN.REG_FC[0], bn=cfg.RCNN.USE_BN))
            iou_branch.append(pt_utils.Conv1d(cfg.RCNN.REG_FC[0], cfg.RCNN.REG_FC[1], bn=cfg.RCNN.USE_BN))
            iou_branch.append(pt_utils.Conv1d(cfg.RCNN.REG_FC[1], 1, activation=None))
            if cfg.RCNN.DP_RATIO >= 0:
                iou_branch.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
            self.iou_branch = nn.Sequential(*iou_branch)
            #pass

        # regression layer
        per_loc_bin_num = int(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel += (1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)

        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)

        self.proposal_target_layer = ProposalTargetLayer()
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, input_data):
        """
        :param input_data: input dict
        :return:
        """
        if cfg.RCNN.ROI_SAMPLE_JIT:
            if self.training:
                with torch.no_grad():
                    target_dict = self.proposal_target_layer(input_data)

                pts_input = torch.cat((target_dict['sampled_pts'], target_dict['pts_feature']), dim=2)
                target_dict['pts_input'] = pts_input
                if cfg.DEEP_RCNN_FUSION:
                    sampled_xy_cor = target_dict['sampled_xy_cor']  ## B,N,2  [B, 64, 512, 2]
                    sampled_xy_cor = sampled_xy_cor.view(sampled_xy_cor.shape[0],
                                                     sampled_xy_cor.shape[1] * sampled_xy_cor.shape[2], 2)
            else:
                rpn_xyz, rpn_features = input_data['rpn_xyz'], input_data['rpn_features']
                batch_rois = input_data['roi_boxes3d']

                pts_extra_input_list = []
                if cfg.DEEP_RCNN_FUSION:
                    pts_extra_input_list.append(input_data['l_xy_cor'])   #########

                if cfg.RCNN.USE_INTENSITY:
                    pts_extra_input_list.append([input_data['rpn_intensity'].unsqueeze(dim=2),
                                                 input_data['seg_mask'].unsqueeze(dim=2)])
                else:
                    pts_extra_input_list.append(input_data['seg_mask'].unsqueeze(dim=2))
                
                # if cfg.RCNN.USE_INTENSITY:
                #     pts_extra_input_list = [input_data['rpn_intensity'].unsqueeze(dim=2),
                #                             input_data['seg_mask'].unsqueeze(dim=2)]
                # else:
                #     pts_extra_input_list = [input_data['seg_mask'].unsqueeze(dim=2)]

                if cfg.RCNN.USE_DEPTH:
                    pts_depth = input_data['pts_depth'] / 70.0 - 0.5
                    pts_extra_input_list.append(pts_depth.unsqueeze(dim=2))
                pts_extra_input = torch.cat(pts_extra_input_list, dim=2)

                pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2)
                pooled_features, pooled_empty_flag = \
                        roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                                      sampled_pt_num=cfg.RCNN.NUM_POINTS)

                if cfg.DEEP_RCNN_FUSION:
                    sampled_pts, sampled_xy_cor, sampled_features = \
                        pooled_features[:, :, :, 0:3], pooled_features[:, :, :, 3:5], pooled_features[:, :, :, 5:]
                    sampled_xy_cor = sampled_xy_cor.view(sampled_pts.shape[0],
                                                         sampled_pts.shape[1] * sampled_pts.shape[2], 2)
                    pooled_features = torch.cat((sampled_pts, sampled_features), dim=-1)

                # canonical transformation
                batch_size = batch_rois.shape[0]
                roi_center = batch_rois[:, :, 0:3]
                pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
                for k in range(batch_size):
                    pooled_features[k, :, :, 0:3] = kitti_utils.rotate_pc_along_y_torch(pooled_features[k, :, :, 0:3],
                                                                                        batch_rois[k, :, 6])

                pts_input = pooled_features.view(-1, pooled_features.shape[2], pooled_features.shape[3])
        else:
            pts_input = input_data['pts_input']
            target_dict = {}
            target_dict['pts_input'] = input_data['pts_input']
            target_dict['roi_boxes3d'] = input_data['roi_boxes3d']
            if self.training:
                target_dict['cls_label'] = input_data['cls_label']
                target_dict['reg_valid_mask'] = input_data['reg_valid_mask']
                target_dict['gt_of_rois'] = input_data['gt_boxes3d_ct']

        xyz, features = self._break_up_pc(pts_input)

        if cfg.RCNN.USE_RPN_FEATURES: ## True
            xyz_input = pts_input[..., 0:self.rcnn_input_channel].transpose(1, 2).unsqueeze(dim=3)
            xyz_feature = self.xyz_up_layer(xyz_input)

            rpn_feature = pts_input[..., self.rcnn_input_channel:].transpose(1, 2).unsqueeze(dim=3)

            merged_feature = torch.cat((xyz_feature, rpn_feature), dim=1)
            merged_feature = self.merge_down_layer(merged_feature)
            l_xyz, l_features = [xyz], [merged_feature.squeeze(dim=3)]
        else:
            l_xyz, l_features = [xyz], [features]


        if cfg.DEEP_RCNN_FUSION:
            batch_size = sampled_xy_cor.shape[0]
            l_xy_cor = [sampled_xy_cor] ## torch.Size([1, 51200, 2])
            img = [input_data['img_feature']] # [1, 32, 384, 1280]


        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_index = self.SA_modules[i](l_xyz[i], l_features[i])

            if cfg.DEEP_RCNN_FUSION & (i < len(self.SA_modules) - 1):  ###
                if cfg.RCNN.SA_CONFIG.NPOINTS[i]==-1:
                    #print("####cfg.RCNN.SA_CONFIG.NPOINTS[i]###:", cfg.RCNN.SA_CONFIG.NPOINTS[i])
                    #print("#######cfg.RCNN.SA_CONFIG.NPOINTS[i]==-1!!!#########")
                    NUM_POINTS = 1
                else:
                    #print("####cfg.RCNN.SA_CONFIG.NPOINTS[i]###:", cfg.RCNN.SA_CONFIG.NPOINTS[i])
                    NUM_POINTS = cfg.RCNN.SA_CONFIG.NPOINTS[i]

                # print('\n')
                #print("#######USE DEEP_RCNN_FUSION!!!#########i=:", i)
                li_index = li_index.view(batch_size, -1)
                li_index = li_index.long().unsqueeze(-1).repeat(1,1,2) ## [1, 12800, 2]
                li_xy_cor = torch.gather(l_xy_cor[i],1,li_index)
                image = self.Img_Block_RCNN[i](img[i])

                if cfg.CROSS_FUSION:
                    img_shape = image.shape
                    cross_feat = li_features.clone()
                    cross_feat = cross_feat.contiguous().view(batch_size, -1, cross_feat.shape[1], NUM_POINTS).permute(0, 2, 1, 3)  # (B,ROIS,C,N)
                    cross_feat = cross_feat.contiguous().view(batch_size, cross_feat.shape[1], -1)
                    project_point2img_feature = grid_sample_reverse(cross_feat, li_xy_cor, img_shape)
                    # print('#######project_point2img_feature:', project_point2img_feature.shape)
                    # print('#######image:', image.shape)
                    image = self.Cross_Fusion[i](project_point2img_feature, image)
                    # l_xy_cor_ori.append(li_xy_cor_ori)

                img_gather_feature = Feature_Gather(image, li_xy_cor)

                img_gather_feature = img_gather_feature.contiguous().view(batch_size,image.shape[1], -1, NUM_POINTS).permute(0, 2, 1, 3)  # [1, 100, 32, 128]
                img_gather_feature = img_gather_feature.contiguous().view(-1, image.shape[1], NUM_POINTS)  # [100, 32, 128]

                li_features = self.Fusion_Conv_RCNN[i](li_features, img_gather_feature)  ## [100, 128, 128]
                l_xy_cor.append(li_xy_cor)  ## [1, 12800, 2]
                img.append(image)

            l_xyz.append(li_xyz)
            l_features.append(li_features)

        rcnn_cls = self.cls_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        if cfg.USE_IOU_BRANCH:
            rcnn_iou_branch = self.iou_branch(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B,1)
            ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg, 'rcnn_iou_branch': rcnn_iou_branch}
        else:
            ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg}

        if self.training:
            ret_dict.update(target_dict)
        return ret_dict