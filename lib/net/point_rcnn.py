import torch
import torch.nn as nn
from lib.net.rpn import RPN
from lib.net.rcnn_net import RCNNNet
from lib.config import cfg

from torch.nn.functional import grid_sample


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


class PointRCNN(nn.Module):
    def __init__(self, num_classes, use_xyz = True, mode = 'TRAIN'):
        super().__init__()

        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz = use_xyz, mode = mode)

        if cfg.RCNN.ENABLED:
            rcnn_input_channels = 128  # channels of rpn features
            if cfg.RCNN.BACKBONE == 'pointnet':
                self.rcnn_net = RCNNNet(num_classes = num_classes, input_channels = rcnn_input_channels,
                                        use_xyz = use_xyz)
            elif cfg.RCNN.BACKBONE == 'pointsift':
                pass
            else:
                raise NotImplementedError

    def forward(self, input_data):

        if cfg.RPN.ENABLED:
            output = { }
            # rpn inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)

                output.update(rpn_output)
                backbone_xyz = rpn_output['backbone_xyz']
                backbone_features = rpn_output['backbone_features']
                ####print('##########xyz.shape:', backbone_xyz.shape)

            # rcnn inference
            if cfg.RCNN.ENABLED:
                with torch.no_grad():
                    rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']

                    rpn_scores_raw = rpn_cls[:, :, 0]

                    if cfg.USE_IMAGE_SCORE:
                        rpn_point_scores = rpn_scores_raw
                        rpn_image_scores = Feature_Gather(rpn_output['rpn_image_seg'], rpn_output['l_xy_cor']).squeeze(1)
                        output['rpn_point_scores'] = rpn_point_scores
                        output['rpn_image_scores'] = rpn_image_scores
                        rpn_scores_raw = (rpn_image_scores + rpn_point_scores)


                    rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                    seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
                    pts_depth = torch.norm(backbone_xyz, p = 2, dim = 2)

                    # proposal layer
                    rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)

                    output['rois'] = rois
                    output['roi_scores_raw'] = roi_scores_raw
                    output['seg_result'] = seg_mask

                rcnn_input_info = { 'rpn_xyz'     : backbone_xyz,
                                    'rpn_features': backbone_features.permute((0, 2, 1)),
                                    'seg_mask'    : seg_mask,
                                    'roi_boxes3d' : rois,
                                    'pts_depth'   : pts_depth
                                    }

                if cfg.DEEP_RCNN_FUSION:
                    rcnn_input_info['img_feature'] = rpn_output['img_feature']
                    rcnn_input_info['l_xy_cor'] = rpn_output['l_xy_cor']


                if self.training:
                    rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']

                rcnn_output = self.rcnn_net(rcnn_input_info)
                output.update(rcnn_output)

        elif cfg.RCNN.ENABLED:
            output = self.rcnn_net(input_data)
        else:
            raise NotImplementedError

        return output

if __name__=='__main__':
    pass