# 3DInvNet
1. Commands for training Inverter: \
python workspace/trainInvNet.py \
--model InvNetModel_ff \
--lossfc MAE_loss \
--batch_size 4 \
--lr 0.001 \
--lr_decay 0.98 \
--max_epoch 100 \
--id ff \
--train_data_path dataset/train/mask1 \
--train_mask_path dataset/train/mask2 \
--test_data_path dataset/test/mask1 \
--test_mask_path dataset/test/mask2 \
--model_path workspace_Inverter/exp/model/ \
--visualization_path workspace_Inverter/exp/visual/ \
--save_model
