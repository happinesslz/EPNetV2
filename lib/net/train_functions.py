import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.utils.loss_utils as loss_utils
from lib.config import cfg
from collections import namedtuple
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

def model_joint_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ['loss', 'tb_dict', 'disp_dict'])
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

    def model_fn(model, data):
        if cfg.RPN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']
            # pts_rgb = data['pts_rgb']

            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']  # (2, 4096)  (2, 4096, 7)
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking = True).long()
                rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking = True).float()

            inputs = torch.from_numpy(pts_input).cuda(non_blocking = True).float()
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking = True).float()
            input_data = { 'pts_input': inputs, 'gt_boxes3d': gt_boxes3d }
        else:
            input_data = { }
            for key, val in data.items():
                if key != 'sample_id':
                    input_data[key] = torch.from_numpy(val).contiguous().cuda(non_blocking = True).float()
            if not cfg.RCNN.ROI_SAMPLE_JIT:
                pts_input = torch.cat((input_data['pts_input'], input_data['pts_features']), dim = -1)
                input_data['pts_input'] = pts_input
        # input()
        if cfg.LI_FUSION.ENABLED:
            img = torch.from_numpy(data['img']).cuda(non_blocking = True).float().permute((0, 3, 1, 2))
            pts_origin_xy = torch.from_numpy(data['pts_origin_xy']).cuda(non_blocking = True).float()
            input_data['img'] = img
            input_data['pts_origin_xy'] = pts_origin_xy  # [2, 4096, 2]
        if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
            pts_rgb = data['rgb']
            # print(pts_rgb.shape)
            pts_rgb = torch.from_numpy(pts_rgb).cuda(non_blocking = True).float()
            input_data['pts_rgb'] = pts_rgb
        if cfg.USE_PAINTING_SCORE:
            pts_paint_scores = data['pts_paint_scores']
            input_data['pts_paint_scores'] = torch.from_numpy(pts_paint_scores).cuda(non_blocking = True).float()
        if cfg.USE_PAINTING_FEAT:
            pts_paint_feats = data['pts_paint_feats']
            input_data['pts_paint_feats'] = torch.from_numpy(pts_paint_feats).cuda(non_blocking = True).float()
        ret_dict = model(input_data)

        tb_dict = { }
        disp_dict = { }
        loss = 0
        if cfg.RPN.ENABLED and not cfg.RPN.FIXED:
            rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']
            if cfg.USE_IMAGE_LOSS:
                #rpn_image_seg_label = rpn_cls_label  # [2, 4096]
                if cfg.USE_IMG_DENSE_LOSS:
                    img_seg_predict = ret_dict['rpn_image_seg']  # [2, 1, 384, 1280]
                    img_seg_sparse_predict = Feature_Gather(img_seg_predict, ret_dict['l_xy_cor'])  # [2, 1, 4096]
                    rpn_image_seg_label = torch.from_numpy(data['img_seg_mask']).cuda(non_blocking=True).float()
                else:
                    rpn_image_seg = ret_dict['rpn_image_seg']  # [2, 1, 384, 1280]
                    img_seg_predict = Feature_Gather(rpn_image_seg, ret_dict['l_xy_cor'])  # [2, 1, 4096]
                    rpn_image_seg_label = torch.from_numpy(data['rpn_cls_label']).cuda(non_blocking = True).float()
                    img_seg_sparse_predict = img_seg_predict

            else:
                rpn_image_seg_label = None
                img_seg_predict = None
                img_seg_sparse_predict = None

            rpn_loss, rpn_loss_cls, rpn_loss_loc, rpn_loss_angle, rpn_loss_size, rpn_loss_iou = get_rpn_loss(model,
                                                                                                             rpn_cls,
                                                                                                             rpn_reg,
                                                                                                             rpn_cls_label,
                                                                                                             rpn_reg_label,
                                                                                                             tb_dict,
                                                                                                             img_seg_predict=img_seg_predict,
                                                                                                             rpn_image_seg_label = rpn_image_seg_label,
                                                                                                             img_seg_sparse_predict = img_seg_sparse_predict
                                                                                                             )
            rpn_loss = rpn_loss * cfg.TRAIN.RPN_TRAIN_WEIGHT
            loss += rpn_loss
            disp_dict['rpn_loss'] = rpn_loss.item()
            disp_dict['rpn_loss_cls'] = rpn_loss_cls.item()
            disp_dict['rpn_loss_loc'] = rpn_loss_loc.item()
            disp_dict['rpn_loss_angle'] = rpn_loss_angle.item()
            disp_dict['rpn_loss_size'] = rpn_loss_size.item()
            disp_dict['rpn_loss_iou'] = rpn_loss_iou.item()
            # if cfg.USE_IMAGE_LOSS:
            #     disp_dict['rpn_loss_iou'] = rpn_loss_iou.item()


        if cfg.RCNN.ENABLED:
            if cfg.USE_IOU_BRANCH:
                rcnn_loss,iou_loss,iou_branch_loss = get_rcnn_loss(model, ret_dict, tb_dict)
                disp_dict['reg_fg_sum'] = tb_dict['rcnn_reg_fg']

                rcnn_loss = rcnn_loss * cfg.TRAIN.RCNN_TRAIN_WEIGHT
                disp_dict['rcnn_loss'] = rcnn_loss.item()
                loss += rcnn_loss
                disp_dict['loss'] = loss.item()
                disp_dict['rcnn_iou_loss'] = iou_loss.item()
                disp_dict['iou_branch_loss'] = iou_branch_loss.item()
            else:
                rcnn_loss = get_rcnn_loss(model, ret_dict, tb_dict)
                disp_dict['reg_fg_sum'] = tb_dict['rcnn_reg_fg']

                rcnn_loss = rcnn_loss * cfg.TRAIN.RCNN_TRAIN_WEIGHT
                disp_dict['rcnn_loss'] = rcnn_loss.item()
                loss += rcnn_loss
                disp_dict['loss'] = loss.item()

        return ModelReturn(loss, tb_dict, disp_dict)

    def get_rpn_loss(model, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict, img_seg_predict=None, rpn_image_seg_label=None,img_seg_sparse_predict=None):
        batch_size = rpn_cls.shape[0]
        if isinstance(model, nn.DataParallel):
            rpn_cls_loss_func = model.module.rpn.rpn_cls_loss_func
            rpn_img_seg_loss_func = model.module.rpn.rpn_img_seg_loss_func
        else:
            rpn_cls_loss_func = model.rpn.rpn_cls_loss_func
            rpn_img_seg_loss_func = model.rpn.rpn_img_seg_loss_func

        rpn_cls_label_flat = rpn_cls_label.view(-1)
        rpn_cls_flat = rpn_cls.view(-1)
        fg_mask = (rpn_cls_label_flat > 0)

        # RPN classification loss
        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls, rpn_cls_label_flat)

        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            rpn_cls_target = (rpn_cls_label_flat > 0).float()
            pos = (rpn_cls_label_flat > 0).float()
            neg = (rpn_cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min = 1.0)
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls_flat, rpn_cls_target, cls_weights)
            rpn_loss_cls_pos = (rpn_loss_cls * pos).sum()
            rpn_loss_cls_neg = (rpn_loss_cls * neg).sum()
            rpn_loss_cls = rpn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rpn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rpn_loss_cls_neg.item()

            if cfg.USE_IMAGE_LOSS:
                ## full image seg loss:
                rpn_image_seg_label = rpn_image_seg_label.view(-1)
                rpn_image_seg_target = (rpn_image_seg_label > 0).float()

                img_pos = (rpn_image_seg_label>0).float()
                img_neg = (rpn_image_seg_label==0).float()
                img_weight = (img_pos + img_neg) / (torch.clamp(img_pos.sum(), min = 1.0))
                rpn_img_seg_loss = rpn_cls_loss_func(img_seg_predict.view(-1), rpn_image_seg_target, img_weight)
                rpn_img_seg_loss = rpn_img_seg_loss.sum()
                # print('#######rpn_img_seg_loss:', rpn_img_seg_loss)

            if cfg.USE_MC_LOSS:
                P1 = F.log_softmax(img_seg_sparse_predict.view(batch_size, -1), dim=-1)  # img prop
                P2 = F.log_softmax(rpn_cls_flat.view(batch_size, -1), dim=-1)  # point prop

                P1_1 = F.softmax(img_seg_sparse_predict.view(batch_size, -1), dim=-1)  # img prop
                P2_1 = F.softmax(rpn_cls_flat.view(batch_size, -1), dim=-1)  # point prop
                P = (P1_1.clone() + P2_1.clone()) / 2.0

                kl_loss_i2p = F.kl_div(P1, P.detach(), reduction='none')
                kl_loss_p2i = F.kl_div(P2, P.detach(), reduction='none')

                mc_loss = cfg.I2P_Weight * kl_loss_i2p + cfg.P2I_Weight * kl_loss_p2i

                p1_score = torch.sigmoid(img_seg_sparse_predict.view(batch_size, -1))
                p2_score = torch.sigmoid(rpn_cls_flat.view(batch_size, -1))

                if cfg.ADD_MC_MASK:
                    kl_element_weight = (torch.max(p1_score, p2_score) >= cfg.MC_MASK_THRES).float()
                else:
                    kl_element_weight = torch.ones_like(p1_score)
                mc_loss = (kl_element_weight.contiguous().view(-1) * mc_loss.contiguous().view(-1) * (pos + neg)).sum()


            if cfg.USE_IMAGE_SCORE:
                # rpn_cls_flat = (rpn_cls_flat + img_seg_predict.view(-1)) / 2.0  #### img score divide 2
                rpn_cls_flat = rpn_cls_flat + img_seg_sparse_predict.view(-1)


        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            weight = rpn_cls_flat.new(rpn_cls_flat.shape[0]).fill_(1.0)
            weight[fg_mask] = cfg.RPN.FG_WEIGHT
            rpn_cls_label_target = (rpn_cls_label_flat > 0).float()
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rpn_cls_flat), rpn_cls_label_target,
                                                    weight=weight, reduction='none')
            cls_valid_mask = (rpn_cls_label_flat >= 0).float()
            rpn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min = 1.0)
        else:
            raise NotImplementedError

        # RPN regression loss
        point_num = rpn_reg.size(0) * rpn_reg.size(1)
        fg_sum = fg_mask.long().sum().item()
        # print('#######rpn average fg_sum:', fg_sum / batch_size)
        if fg_sum != 0:
            loss_loc, loss_angle, loss_size, loss_iou, reg_loss_dict = \
                loss_utils.get_reg_loss(torch.sigmoid(rpn_cls_flat)[fg_mask], torch.sigmoid(rpn_cls_flat)[fg_mask],
                                        rpn_reg.view(point_num, -1)[fg_mask],
                                        rpn_reg_label.view(point_num, 7)[fg_mask],
                                        loc_scope = cfg.RPN.LOC_SCOPE,
                                        loc_bin_size = cfg.RPN.LOC_BIN_SIZE,
                                        num_head_bin = cfg.RPN.NUM_HEAD_BIN,
                                        anchor_size = MEAN_SIZE,
                                        get_xz_fine = cfg.RPN.LOC_XZ_FINE,
                                        use_cls_score = True,
                                        use_mask_score = False)

            loss_size = 3 * loss_size  # consistent with old codes
            # print('#######rpn loss_iou:', loss_iou)
            loss_iou = cfg.TRAIN.RPN_CE_WEIGHT * loss_iou
            rpn_loss_reg = loss_loc + loss_angle + loss_size + loss_iou
        else:
            # loss_loc = loss_angle = loss_size = rpn_loss_reg = rpn_loss_cls * 0
            loss_loc = loss_angle = loss_size = loss_iou = rpn_loss_reg = rpn_loss_cls * 0

        rpn_loss = rpn_loss_cls * cfg.RPN.LOSS_WEIGHT[0] + rpn_loss_reg * cfg.RPN.LOSS_WEIGHT[1]
        if cfg.USE_IMAGE_LOSS:
            rpn_loss = rpn_loss + cfg.IMAGE_WEIGHT * rpn_img_seg_loss
            tb_dict['rpn_img_seg_loss'] = rpn_img_seg_loss.item()
        if cfg.USE_MC_LOSS:
            rpn_loss = rpn_loss + cfg.MC_LOSS_WEIGHT * mc_loss
            tb_dict['mc_loss'] = mc_loss.item()


        tb_dict.update({ 'rpn_loss_cls'  : rpn_loss_cls.item(), 'rpn_loss_reg': rpn_loss_reg.item(),
                         'rpn_loss'      : rpn_loss.item(), 'rpn_fg_sum': fg_sum, 'rpn_loss_loc': loss_loc.item(),
                         'rpn_loss_angle': loss_angle.item(), 'rpn_loss_size': loss_size.item(),
                         'rpn_loss_iou'  : loss_iou.item() })

        # return rpn_loss
        return rpn_loss, rpn_loss_cls, loss_loc, loss_angle, loss_size, loss_iou

    def get_rcnn_loss(model, ret_dict, tb_dict):
        rcnn_cls, rcnn_reg = ret_dict['rcnn_cls'], ret_dict['rcnn_reg']
        cls_label = ret_dict['cls_label'].float()
        reg_valid_mask = ret_dict['reg_valid_mask']
        roi_boxes3d = ret_dict['roi_boxes3d']
        roi_size = roi_boxes3d[:, 3:6]
        gt_boxes3d_ct = ret_dict['gt_of_rois']
        pts_input = ret_dict['pts_input']
        mask_score = ret_dict['mask_score']

        gt_iou_weight = ret_dict['gt_iou']

        # rcnn classification loss
        if isinstance(model, nn.DataParallel):
            cls_loss_func = model.module.rcnn_net.cls_loss_func
        else:
            cls_loss_func = model.rcnn_net.cls_loss_func

        cls_label_flat = cls_label.view(-1)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            rcnn_cls_flat = rcnn_cls.view(-1)

            cls_target = (cls_label_flat > 0).float()
            pos = (cls_label_flat > 0).float()
            neg = (cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min = 1.0)

            rcnn_loss_cls = cls_loss_func(rcnn_cls_flat, cls_target, cls_weights)
            rcnn_loss_cls_pos = (rcnn_loss_cls * pos).sum()
            rcnn_loss_cls_neg = (rcnn_loss_cls * neg).sum()
            rcnn_loss_cls = rcnn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rcnn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rcnn_loss_cls_neg.item()

        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), cls_label, reduction = 'none')
            cls_valid_mask = (cls_label_flat >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min = 1.0)

        elif cfg.TRAIN.LOSS_CLS == 'CrossEntropy':
            rcnn_cls_reshape = rcnn_cls.view(rcnn_cls.shape[0], -1)
            cls_target = cls_label_flat.long()
            cls_valid_mask = (cls_label_flat >= 0).float()

            batch_loss_cls = cls_loss_func(rcnn_cls_reshape, cls_target)
            normalizer = torch.clamp(cls_valid_mask.sum(), min = 1.0)
            rcnn_loss_cls = (batch_loss_cls.mean(dim = 1) * cls_valid_mask).sum() / normalizer

        else:
            raise NotImplementedError

        # rcnn regression loss
        batch_size = pts_input.shape[0]
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            # print('#######rcnn average fg_sum:', fg_sum / batch_size)
            if cfg.USE_IOU_BRANCH:
                iou_branch_pred = ret_dict['rcnn_iou_branch']
                iou_branch_pred_fg_mask = iou_branch_pred[fg_mask]
            else:
                iou_branch_pred_fg_mask = None

            all_anchor_size = roi_size
            anchor_size = all_anchor_size[fg_mask] if cfg.RCNN.SIZE_RES_ON_ROI else MEAN_SIZE

            loss_loc, loss_angle, loss_size, loss_iou, reg_loss_dict = \
                loss_utils.get_reg_loss(torch.sigmoid(rcnn_cls_flat)[fg_mask], mask_score[fg_mask],
                                        rcnn_reg.view(batch_size, -1)[fg_mask],
                                        gt_boxes3d_ct.view(batch_size, 7)[fg_mask],
                                        loc_scope = cfg.RCNN.LOC_SCOPE,
                                        loc_bin_size = cfg.RCNN.LOC_BIN_SIZE,
                                        num_head_bin = cfg.RCNN.NUM_HEAD_BIN,
                                        anchor_size = anchor_size,
                                        get_xz_fine = True, get_y_by_bin = cfg.RCNN.LOC_Y_BY_BIN,
                                        loc_y_scope = cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size = cfg.RCNN.LOC_Y_BIN_SIZE,
                                        get_ry_fine = True,
                                        use_cls_score = True,
                                        use_mask_score = True,
                                        gt_iou_weight = gt_iou_weight[fg_mask],
                                        use_iou_branch = cfg.USE_IOU_BRANCH,
                                        iou_branch_pred = iou_branch_pred_fg_mask)

            loss_size = 3 * loss_size  # consistent with old codes
            # rcnn_loss_reg = loss_loc + loss_angle + loss_size
            # print('#######rcnn loss_iou:', loss_iou)
            loss_iou = cfg.TRAIN.CE_WEIGHT * loss_iou
            if cfg.USE_IOU_BRANCH:
                iou_branch_loss = reg_loss_dict['iou_branch_loss']
                rcnn_loss_reg = loss_loc + loss_angle + loss_size + loss_iou  + iou_branch_loss
            else:
                rcnn_loss_reg = loss_loc + loss_angle + loss_size + loss_iou
            tb_dict.update(reg_loss_dict)
        else:
            loss_loc = loss_angle = loss_size = loss_iou = rcnn_loss_reg = iou_branch_loss = rcnn_loss_cls * 0

        rcnn_loss = rcnn_loss_cls + rcnn_loss_reg
        tb_dict['rcnn_loss_cls'] = rcnn_loss_cls.item()
        tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
        tb_dict['rcnn_loss'] = rcnn_loss.item()

        tb_dict['rcnn_loss_loc'] = loss_loc.item()
        tb_dict['rcnn_loss_angle'] = loss_angle.item()
        tb_dict['rcnn_loss_size'] = loss_size.item()
        tb_dict['rcnn_loss_iou'] = loss_iou.item()
        tb_dict['rcnn_cls_fg'] = (cls_label > 0).sum().item()
        tb_dict['rcnn_cls_bg'] = (cls_label == 0).sum().item()
        tb_dict['rcnn_reg_fg'] = reg_valid_mask.sum().item()

        if cfg.USE_IOU_BRANCH:
            tb_dict['iou_branch_loss'] = iou_branch_loss.item()
            # print('\n')
            # print('iou_branch_loss:',iou_branch_loss.item())
            return rcnn_loss, loss_iou, iou_branch_loss
        else:
            return rcnn_loss


    return model_fn
