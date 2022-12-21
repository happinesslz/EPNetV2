#! /bin/bash


################################## train CYC
CUDA_VISIBLE_DEVICES=2,3 python train_rcnn.py --cfg_file cfgs/CYC_EPNet_plus_plus.yaml \
--batch_size 4 --train_mode rcnn_online --epochs 50 --mgpus --ckpt_save_interval 1 \
--output_dir ./log/CYC_EPNet_plus_plus_js_mask_thr_02/   \
--data_path ../data/ \
--set LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2 RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH True TRAIN.CE_WEIGHT 5.0 CROSS_FUSION True \
USE_SELF_ATTENTION False DEEP_RCNN_FUSION False USE_IMAGE_LOSS True IMAGE_WEIGHT 1.0 USE_IMAGE_SCORE True USE_IMG_DENSE_LOSS True USE_KL_LOSS True  USE_KL_LOSS_TYPE 'JS'  \
KL_LOSS_WEIGHT 1.0 USE_P2I_GATE True  TRAIN.RPN_CE_WEIGHT 5.0  SAVE_MODEL_PREP 0.8  I2P_Weight 0.5 P2I_Weight 0.5  ADD_JS_MASK True JS_MASK_THRES 0.2



################################## eval CYC
CUDA_VISIBLE_DEVICES=2 python eval_rcnn.py --cfg_file cfgs/CYC_EPNet_plus_plus.yaml --eval_mode rcnn_online  \
--eval_all  --output_dir ./log/CYC_EPNet_plus_plus_js_mask_thr_02/eval_results/ \
--data_path ../data/ \
--ckpt_dir ./log/CYC_EPNet_plus_plus_js_mask_thr_02/ckpt \
--set LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2 RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH True TRAIN.CE_WEIGHT 5.0 CROSS_FUSION True \
USE_SELF_ATTENTION False DEEP_RCNN_FUSION False USE_IMAGE_LOSS True IMAGE_WEIGHT 1.0 USE_IMAGE_SCORE True USE_IMG_DENSE_LOSS True USE_KL_LOSS True  USE_KL_LOSS_TYPE 'JS'  \
KL_LOSS_WEIGHT 1.0 USE_P2I_GATE True  TRAIN.RPN_CE_WEIGHT 5.0  SAVE_MODEL_PREP 0.8  I2P_Weight 0.5 P2I_Weight 0.5  ADD_JS_MASK True JS_MASK_THRES 0.2



#CUDA_VISIBLE_DEVICES=2 python eval_rcnn.py --cfg_file cfgs/CYC_EPNet_plus_plus.yaml --eval_mode rcnn_online  \
#--eval_all  --output_dir ./log/CYC_EPNet_plus_plus_js_mask_thr_01/eval_results_200_08/ \
#--data_path ../data/ \
#--ckpt_dir ./log/CYC_EPNet_plus_plus_js_mask_thr_01/ckpt \
#--set LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2 RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH True TRAIN.CE_WEIGHT 5.0 CROSS_FUSION True \
#USE_SELF_ATTENTION False DEEP_RCNN_FUSION False USE_IMAGE_LOSS True IMAGE_WEIGHT 1.0 USE_IMAGE_SCORE True USE_IMG_DENSE_LOSS True USE_KL_LOSS True  USE_KL_LOSS_TYPE 'JS'  \
#KL_LOSS_WEIGHT 1.0 USE_P2I_GATE True  TRAIN.RPN_CE_WEIGHT 5.0  SAVE_MODEL_PREP 0.8  I2P_Weight 0.5 P2I_Weight 0.5  ADD_JS_MASK True JS_MASK_THRES 0.1 TEST.RPN_POST_NMS_TOP_N 200
#
#
#
#CUDA_VISIBLE_DEVICES=2 python eval_rcnn.py --cfg_file cfgs/CYC_EPNet_plus_plus.yaml --eval_mode rcnn_online  \
#--eval_all  --output_dir ./log/CYC_EPNet_plus_plus_js_mask_thr_01/eval_results_200_08/ \
#--data_path ../data/ \
#--ckpt_dir ./log/CYC_EPNet_plus_plus_js_mask_thr_01/ckpt \
#--set LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2 RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH True TRAIN.CE_WEIGHT 5.0 CROSS_FUSION True \
#USE_SELF_ATTENTION False DEEP_RCNN_FUSION False USE_IMAGE_LOSS True IMAGE_WEIGHT 1.0 USE_IMAGE_SCORE True USE_IMG_DENSE_LOSS True USE_KL_LOSS True  USE_KL_LOSS_TYPE 'JS'  \
#KL_LOSS_WEIGHT 1.0 USE_P2I_GATE True  TRAIN.RPN_CE_WEIGHT 5.0  SAVE_MODEL_PREP 0.8  I2P_Weight 0.5 P2I_Weight 0.5  ADD_JS_MASK True JS_MASK_THRES 0.1 TEST.RPN_POST_NMS_TOP_N 200
#
#
