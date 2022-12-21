#! /bin/bash


################################## train CAR
CUDA_VISIBLE_DEVICES=2,3 python train_rcnn.py --cfg_file cfgs/CAR_EPNet_plus_plus.yaml \
--batch_size 4 --train_mode rcnn_online --epochs 50 --mgpus --ckpt_save_interval 1 \
--output_dir ./log/CAR_EPNet_plus_plus_only_cb_fusion_run2/   \
--data_path ../data/ \
--set LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2 \
 USE_IOU_BRANCH True TRAIN.CE_WEIGHT 5.0 CROSS_FUSION True USE_P2I_GATE True

################################## eval CAR
#CUDA_VISIBLE_DEVICES=2 python eval_rcnn.py --cfg_file cfgs/CAR_EPNet_plus_plus.yaml --eval_mode rcnn_online  \
#--eval_all  --output_dir ./log/CAR_EPNet_plus_plus_only_cb_fusion_run2/eval_results/ \
#--data_path ../data/ --ckpt_dir ./log/CAR_EPNet_plus_plus_only_cb_fusion_run2/ckpt \
#--set LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2 \
# USE_IOU_BRANCH True TRAIN.CE_WEIGHT 5.0 CROSS_FUSION True USE_P2I_GATE True

