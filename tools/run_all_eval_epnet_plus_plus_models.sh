#! /bin/bash

################################## eval CAR
CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/CAR_EPNet_plus_plus.yaml --eval_mode rcnn_online  \
--output_dir ./epnet_plus_plus_released_trained_models/CAR/eval_results/ \
--data_path ../data/ --ckpt ./epnet_plus_plus_released_trained_models/CAR/checkpoint_epoch_43.pth \
--set LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True  CROSS_FUSION True USE_P2I_GATE True \
DEEP_RCNN_FUSION False USE_IMAGE_LOSS True IMAGE_WEIGHT 1.0 USE_IMAGE_SCORE True


################################## eval PED
CUDA_VISIBLE_DEVICES=1 python eval_rcnn.py --cfg_file cfgs/PED_EPNet_plus_plus.yaml --eval_mode rcnn_online  \
--output_dir ./epnet_plus_plus_released_trained_models/PED/eval_results/ \
--data_path ../data/ --ckpt ./epnet_plus_plus_released_trained_models/PED/checkpoint_epoch_44.pth \
--set LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True  CROSS_FUSION True USE_P2I_GATE True \
DEEP_RCNN_FUSION False USE_IMAGE_LOSS True IMAGE_WEIGHT 1.0 USE_IMAGE_SCORE True

################################## eval CYC
CUDA_VISIBLE_DEVICES=2 python eval_rcnn.py --cfg_file cfgs/CYC_EPNet_plus_plus.yaml --eval_mode rcnn_online  \
--output_dir ./epnet_plus_plus_released_trained_models/CYC/eval_results/ \
--data_path ../data/ --ckpt ./epnet_plus_plus_released_trained_models/CYC/checkpoint_epoch_50.pth \
--set LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True  CROSS_FUSION True USE_P2I_GATE True \
DEEP_RCNN_FUSION False USE_IMAGE_LOSS True IMAGE_WEIGHT 1.0 USE_IMAGE_SCORE True
