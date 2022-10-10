# 3DInvNet
Implementation codes for the paper "3DInvNet: A Deep Learning-Based 3D Ground-Penetrating Radar Data Inversion"
1. Commands for training Inverter: \
python workspace_Inverter/trainInvNet.py \
--model InvNetModel_ff \
--lossfc MAE_loss \
--batch_size 4 \
--lr 0.001 \
--lr_decay 0.98 \
--max_epoch 100 \
--id MAE \
--train_data_path dataset/train/mask1 \
--train_mask_path dataset/train/mask2 \
--test_data_path dataset/test/mask1 \
--test_mask_path dataset/test/mask2 \
--model_path workspace_Inverter/exp/model/ \
--visualization_path workspace_Inverter/exp/visual/ \
--save_model

2. Commands for training Denoiser: \
python workspace_Denoiser/trainDenoisingNet.py \
--model DenoisingNetModel \
--lossfc MSE_loss \
--batch_size 4 \
--lr 0.001 \
--lr_decay 0.98 \
--max_epoch 100 \
--id MSE \
--train_data_path dataset/train/data \
--train_mask_path dataset/train/mask1 \
--test_data_path dataset/test/data \
--test_mask_path dataset/test/mask1 \
--model_path workspace_Denoiser/exp/model/ \
--visualization_path workspace_Denoiser/exp/visual/ \
--save_model
