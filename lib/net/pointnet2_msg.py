import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from lib.config import cfg
from torch.nn.functional import grid_sample

from lib.utils.sample2grid import sample2grid_F,sample2GaussianGrid_F, sample2BilinearGrid_F
from lib.net.self_attention import PointContext3D


BatchNorm2d = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)

def conv1x1(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride,
                     padding = 0, bias = False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(outplanes, outplanes, 2*stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out

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


class Fusion_Cross_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):

        super(Fusion_Cross_Conv, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.conv1 = conv3x3(inplanes, outplanes, stride=1)
        self.bn1 = BatchNorm2d(outplanes)

    def forward(self, point_features, img_features):
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


class P2IA_Layer(nn.Module):
    def __init__(self, channels):
        print('##############ADDITION PI2 ATTENTION#########')
        super(P2IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.ic // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.pc, self.pc, 1),
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

        point_feas_new = self.conv1(point_feas)
        out = point_feas_new * att

        return out


class Fusion_Cross_Conv_Gate(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        print('##############USE Fusion_Cross_Conv_Gate(ADD)#########')
        super(Fusion_Cross_Conv_Gate, self).__init__()
        self.P2IA_Layer = P2IA_Layer(channels=[inplanes_I, inplanes_P])
        self.inplanes = inplanes_I + inplanes_P
        self.outplanes = outplanes
        self.conv1 = conv3x3(self.inplanes, self.outplanes, stride=1)
        self.bn1 = BatchNorm2d(self.outplanes)

    def forward(self, point_features, img_features, li_xy_cor, image):

        point_features = self.P2IA_Layer(img_features, point_features)

        project_point2img_feature = grid_sample_reverse(point_features, li_xy_cor, img_shape=image.shape)

        fusion_features = torch.cat([project_point2img_feature, image], dim=1)

        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


class IA_Layer(nn.Module):
    def __init__(self, channels):
        super(IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),  #####
                                    nn.BatchNorm1d(self.pc),  ####
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
        att = F.sigmoid(self.fc3(F.tanh(ri + rp))) # BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1) # B1N
        # print(img_feas.size(), att.size())

        img_feas_new = self.conv1(img_feas)
        out = img_feas_new * att

        return out


class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv, self).__init__()

        self.IA_Layer = IA_Layer(channels = [inplanes_I, inplanes_P])
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        img_features = self.IA_Layer(img_features, point_features)

        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


def Feature_Gather(feature_map, xy):
    """
    :param xy:(B,N,2)  normalize to [-1,1]
    :param feature_map:(B,C,H,W)
    :return:
    """

    # use grid_sample for this.
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)

    interpolate_feature = grid_sample(feature_map, xy)  # (B,C,1,N)

    return interpolate_feature.squeeze(2) # (B,C,N)


def grid_sample_reverse(point_feature, xy, img_shape):

    # print('#######point_feature:', point_feature.shape)
    # print('#######xy:', xy.shape)
    # print('#######size:', size)
    size = [i for i in img_shape]
    size[1] = point_feature.shape[1]
    project_point2img = sample2BilinearGrid_F(point_feature, xy, size)

    return project_point2img


def get_model(input_channels = 6, use_xyz = True):
    return Pointnet2MSG(input_channels = input_channels, use_xyz = use_xyz)


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels = 6, use_xyz = True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(cfg.RPN.SA_CONFIG.NPOINTS.__len__()):
            mlps = cfg.RPN.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            # if cfg.USE_SELF_ATTENTION:
            #     channel_out += cfg.RPN.SA_CONFIG.ATTN[k]

            self.SA_modules.append(
                    PointnetSAModuleMSG(
                            npoint = cfg.RPN.SA_CONFIG.NPOINTS[k],
                            radii = cfg.RPN.SA_CONFIG.RADIUS[k],
                            nsamples = cfg.RPN.SA_CONFIG.NSAMPLE[k],
                            mlps = mlps,
                            use_xyz = use_xyz,
                            bn = cfg.RPN.USE_BN
                    )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        ##################
        if cfg.LI_FUSION.ENABLED:
            self.Img_Block = nn.ModuleList()
            self.Fusion_Conv = nn.ModuleList()
            self.DeConv = nn.ModuleList()
            if cfg.CROSS_FUSION:
                self.Cross_Fusion = nn.ModuleList()
            if cfg.USE_IM_DEPTH:
                cfg.LI_FUSION.IMG_CHANNELS[0] = cfg.LI_FUSION.IMG_CHANNELS[0] + 1

            if cfg.INPUT_CROSS_FUSION:
                cfg.LI_FUSION.IMG_CHANNELS[0] = cfg.LI_FUSION.IMG_CHANNELS[0] + 4

            for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
                self.Img_Block.append(BasicBlock(cfg.LI_FUSION.IMG_CHANNELS[i], cfg.LI_FUSION.IMG_CHANNELS[i+1], stride=1))
                if cfg.LI_FUSION.ADD_Image_Attention:
                    self.Fusion_Conv.append(
                            Atten_Fusion_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.POINT_CHANNELS[i],
                                              cfg.LI_FUSION.POINT_CHANNELS[i]))
                else:
                    self.Fusion_Conv.append(Fusion_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1] + cfg.LI_FUSION.POINT_CHANNELS[i],
                                                            cfg.LI_FUSION.POINT_CHANNELS[i]))

                if cfg.CROSS_FUSION:
                    if cfg.USE_P2I_GATE:
                        self.Cross_Fusion.append(Fusion_Cross_Conv_Gate(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.POINT_CHANNELS[i],
                                              cfg.LI_FUSION.IMG_CHANNELS[i + 1]))
                    else:
                        self.Cross_Fusion.append(Fusion_Cross_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1] + cfg.LI_FUSION.POINT_CHANNELS[i],
                                              cfg.LI_FUSION.IMG_CHANNELS[i + 1]))

                self.DeConv.append(nn.ConvTranspose2d(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.DeConv_Reduce[i],
                                                  kernel_size=cfg.LI_FUSION.DeConv_Kernels[i],
                                                  stride=cfg.LI_FUSION.DeConv_Kernels[i]))

            self.image_fusion_conv = nn.Conv2d(sum(cfg.LI_FUSION.DeConv_Reduce), cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, kernel_size = 1)
            self.image_fusion_bn = torch.nn.BatchNorm2d(cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4)

            if cfg.LI_FUSION.ADD_Image_Attention:
                self.final_fusion_img_point = Atten_Fusion_Conv(cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, cfg.LI_FUSION.IMG_FEATURES_CHANNEL, cfg.LI_FUSION.IMG_FEATURES_CHANNEL)
            else:
                self.final_fusion_img_point = Fusion_Conv(cfg.LI_FUSION.IMG_FEATURES_CHANNEL + cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, cfg.LI_FUSION.IMG_FEATURES_CHANNEL)

        if cfg.USE_SELF_ATTENTION: ## set as False
            # ref: https://github.com/AutoVision-cloud/SA-Det3D/blob/main/src/models/backbones_3d/pointnet2_backbone.py
            # point-fsa from cfe
            print('##################USE_SELF_ATTENTION!!!!!!!! ')
            self.context_conv3 = PointContext3D(cfg.RPN.SA_CONFIG, IN_DIM=cfg.RPN.SA_CONFIG.MLPS[2][0][-1] + cfg.RPN.SA_CONFIG.MLPS[2][1][-1])
            self.context_conv4 = PointContext3D(cfg.RPN.SA_CONFIG, IN_DIM=cfg.RPN.SA_CONFIG.MLPS[3][0][-1] + cfg.RPN.SA_CONFIG.MLPS[3][1][-1])
            self.context_fusion_3 = Fusion_Conv(cfg.RPN.SA_CONFIG.ATTN[2] + cfg.RPN.SA_CONFIG.MLPS[2][0][-1] + cfg.RPN.SA_CONFIG.MLPS[2][1][-1],
                                                            cfg.RPN.SA_CONFIG.MLPS[2][0][-1] + cfg.RPN.SA_CONFIG.MLPS[2][1][-1] )
            self.context_fusion_4 = Fusion_Conv(cfg.RPN.SA_CONFIG.ATTN[3] + cfg.RPN.SA_CONFIG.MLPS[3][0][-1] + cfg.RPN.SA_CONFIG.MLPS[3][1][-1],
                                                            cfg.RPN.SA_CONFIG.MLPS[3][0][-1] + cfg.RPN.SA_CONFIG.MLPS[3][1][-1])

        self.FP_modules = nn.ModuleList()

        for k in range(cfg.RPN.FP_MLPS.__len__()):
            pre_channel = cfg.RPN.FP_MLPS[k + 1][-1] if k + 1 < len(cfg.RPN.FP_MLPS) else channel_out
            self.FP_modules.append(
                    PointnetFPModule(mlp = [pre_channel + skip_channel_list[k]] + cfg.RPN.FP_MLPS[k])
            )
        #self.Cross_Fusion_Final = Fusion_Cross_Conv(cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4 + cfg.LI_FUSION.IMG_FEATURES_CHANNEL, cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features


    def forward(self, pointcloud: torch.cuda.FloatTensor, image=None, xy=None):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        batch_size = xyz.shape[0]

        if cfg.LI_FUSION.ENABLED:
            #### normalize xy to [-1,1]
            size_range = [1280.0, 384.0]

            x = xy[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
            y = xy[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0
            xy = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)],dim=-1)
            l_xy_cor = [xy]
            img = [image]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_index = self.SA_modules[i](l_xyz[i], l_features[i])


            if cfg.LI_FUSION.ENABLED:
                li_index = li_index.long().unsqueeze(-1).repeat(1,1,2)
                li_xy_cor = torch.gather(l_xy_cor[i],1,li_index)

                image = self.Img_Block[i](img[i])

                if cfg.CROSS_FUSION:
                    if cfg.USE_P2I_GATE:
                        first_img_gather_feature = Feature_Gather(image, li_xy_cor)  # , scale= 2**(i+1))
                        image = self.Cross_Fusion[i](li_features, first_img_gather_feature, li_xy_cor, image)
                    else:
                        img_shape = image.shape
                        project_point2img_feature = grid_sample_reverse(li_features, li_xy_cor, img_shape)
                        image = self.Cross_Fusion[i](project_point2img_feature, image)

                #print(image.shape)
                img_gather_feature = Feature_Gather(image, li_xy_cor) #, scale= 2**(i+1))

                li_features = self.Fusion_Conv[i](li_features, img_gather_feature)

                if cfg.USE_SELF_ATTENTION:
                    if i == 2:
                        # Get context visa self-attention
                        l_context_3 = self.context_conv3(batch_size, li_features, li_xyz)
                        # Concatenate
                        #li_features = torch.cat([li_features, l_context_3], dim=1)
                        li_features = self.context_fusion_3(li_features, l_context_3)
                    if i == 3:
                        # Get context via self-attention
                        l_context_4 = self.context_conv4(batch_size, li_features, li_xyz)
                        # Concatenate
                        #li_features = torch.cat([li_features, l_context_4], dim=1)
                        li_features = self.context_fusion_4(li_features, l_context_4)

                l_xy_cor.append(li_xy_cor)
                img.append(image)

            l_xyz.append(li_xyz)
            l_features.append(li_features)


        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        if cfg.LI_FUSION.ENABLED:
            DeConv = []
            for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
                DeConv.append(self.DeConv[i](img[i + 1]))
            de_concat = torch.cat(DeConv,dim=1)

            img_fusion = F.relu(self.image_fusion_bn(self.image_fusion_conv(de_concat)))
            img_fusion_gather_feature = Feature_Gather(img_fusion, xy)
            l_features[0] = self.final_fusion_img_point(l_features[0], img_fusion_gather_feature)

        if cfg.LI_FUSION.ENABLED:
            return l_xyz[0], l_features[0], img_fusion, l_xy_cor[0]
        else:
            return l_xyz[0], l_features[0], None, None

class Pointnet2MSG_returnMiddleStages(Pointnet2MSG):
    def __init__(self, input_channels = 6, use_xyz = True):
        super().__init__(input_channels, use_xyz)

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        idxs = []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            idxs.append(idx)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_xyz, l_features, idxs
