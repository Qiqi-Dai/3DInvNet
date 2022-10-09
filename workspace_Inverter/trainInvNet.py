import sys, time, os, tqdm, torch, argparse, warnings, glob
# from torchsummary import summary
import importlib
import scipy.io as sio

from torch.utils.data import DataLoader
from data_loader import DatasetLoader

from InvNet import InvNet

def main():
	warnings.filterwarnings("ignore")

	parser = argparse.ArgumentParser(description = "InvNet Training")

	# Training related parameters setting
	parser.add_argument('--model', type=str, default="InvNetModel", help='The network structure')
	parser.add_argument('--lossfc', type=str, default="MAE_loss", help='The lossfunction')
	parser.add_argument('--batch_size', type=int, default=30, help='Select {} videos as one batch')
	parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
	parser.add_argument('--lr_decay', type=float, default=0.99, help='Learning rate decay rate')
	parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epochs')
	parser.add_argument('--nDataLoaderThread', type=int, default=1, help='Number of loader threads')

	parser.add_argument('--id', type=str, default="test", help='The name for saving outputs')
	parser.add_argument('--pretrain', type=str, default="test", help='The name of thr pre-trained model')

	parser.add_argument('--train_data_path', type=str, default="dataset/train/data",   help='Train Data path')
	parser.add_argument('--train_mask_path', type=str, default="dataset/train/mask",   help='Train Mask path')
	parser.add_argument('--test_data_path', type=str, default="dataset/test/data",   help='Test Data path')
	parser.add_argument('--test_mask_path', type=str, default="dataset/test/mask",   help='Test Mask path')

	# Model save path and visualization result path
	parser.add_argument('--model_path', type=str, default="workspace/exp/model/",   help='Model save path load model path')
	parser.add_argument('--visualization_path', type=str, default="workspace/exp/visual/",   help='Output Visualization Path')
	
	# Mode setting 
	parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')
	parser.add_argument('--load_pretrain', dest='load_pretrain', action='store_true', help='Load pretrain model or not')
	parser.add_argument('--save_model', dest='save_model', action='store_true', help='Save the model or not')
	parser.add_argument('--visual', dest='visual', action='store_true', help='Save the test result')

	## ===== ===== ===== ===== ===== ===== ===== =====
	## Main code
	## ===== ===== ===== ===== ===== ===== ===== =====
	args = parser.parse_args()
	args.visualization_path = args.visualization_path + args.id
	print(args.visualization_path)
	os.makedirs(args.visualization_path, exist_ok = True)

	s = InvNet(**vars(args))
	print("Model para number = %.2f"%(sum(param.numel() for param in s.parameters()) / 1024 / 1024))

	train_Loader = DatasetLoader(data = args.train_data_path, mask = args.train_mask_path, **vars(args))
	train_loader = DataLoader(train_Loader, batch_size = args.batch_size, shuffle = True, num_workers = args.nDataLoaderThread, pin_memory = False)
	
	val_Loader = DatasetLoader(data = args.test_data_path, mask = args.test_mask_path, **vars(args))
	val_loader = DataLoader(val_Loader, batch_size = args.batch_size, shuffle = False, num_workers = args.nDataLoaderThread, pin_memory = False)


	it = 1
	decay = 0
	if args.load_pretrain == True:
		modelfiles = args.model_path + args.pretrain + "/model.model"
		print(modelfiles)
		s.loadParameters(modelfiles)
		print("Model %s loaded from previous state!"%modelfiles)

	clr = s.updateLearningRate(1)
	for ii in range(0, decay):
		clr = s.updateLearningRate(args.lr_decay)

	if args.eval == True:
		s.validate_network(it = it, loader = val_loader, **vars(args))
		quit()

	if args.visual == True:
		s.visualization(data = args.test_data_path, path = args.visualization_path, **vars(args))
		quit()

	loss_set = []
	val_loss_set = []
	while(1):
		loss = s.train_network(it = it, learningrate = max(clr), loader = train_loader, **vars(args))
		val_loss = s.validate_network(it = it, loader = val_loader, **vars(args))

		loss_set.append(loss)
		val_loss_set.append(val_loss)
		if len(loss_set) >= 2:
			if loss_set[-1]> loss_set[-2]:
				clr = s.updateLearningRate(args.lr_decay)
				decay += 1

		if args.save_model == True and val_loss == min(val_loss_set):
			s.saveParameters(args.model_path + args.id + "/model.model")
			s.visualization(data = args.test_data_path, path = args.visualization_path, **vars(args))

		if it >= args.max_epoch or max(clr) == 0:
			quit()

		it += 1

if __name__ == '__main__':
	main()
