#! /bin/bash


################################## train CYC
CUDA_VISIBLE_DEVICES=0,1 python train_rcnn.py --cfg_file cfgs/CYC_EPNet_plus_plus.yaml \
--batch_size 4 --train_mode rcnn_online --epochs 50 --mgpus --ckpt_save_interval 1 \
--output_dir ./log/CYC_EPNet_plus_plus/   \
--data_path ../data/ \
--set LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True  CROSS_FUSION True USE_P2I_GATE True \
DEEP_RCNN_FUSION False USE_IMAGE_LOSS True IMAGE_WEIGHT 1.0 USE_IMAGE_SCORE True USE_IMG_DENSE_LOSS True USE_MC_LOSS True  \
MC_LOSS_WEIGHT 1.0   I2P_Weight 0.5 P2I_Weight 0.5  ADD_MC_MASK True MC_MASK_THRES 0.2 SAVE_MODEL_PREP 0.8


################################## eval CYC
CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/CYC_EPNet_plus_plus.yaml --eval_mode rcnn_online  \
--eval_all  --output_dir ./log/CYC_EPNet_plus_plus/eval_results/ \
--data_path ../data/ \
--ckpt_dir ./log/CYC_EPNet_plus_plus/ckpt \
--set LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True  CROSS_FUSION True USE_P2I_GATE True \
DEEP_RCNN_FUSION False USE_IMAGE_LOSS True IMAGE_WEIGHT 1.0 USE_IMAGE_SCORE True USE_IMG_DENSE_LOSS True USE_MC_LOSS True  \
MC_LOSS_WEIGHT 1.0   I2P_Weight 0.5 P2I_Weight 0.5  ADD_MC_MASK True MC_MASK_THRES 0.2 SAVE_MODEL_PREP 0.8

