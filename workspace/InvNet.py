import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib, sys, time, tqdm, numpy, os, glob
import matplotlib.image as mpimg
import scipy.io as sio

class InvNet(nn.Module):
	## ===== ===== ===== ===== ===== ===== ===== =====
	## Init network
	## ===== ===== ===== ===== ===== ===== ===== =====
	def __init__(self, batch_size, lr = 0.001, model = "InvNetModel", lossfc = "MSE_loss", **kwargs):
		super(InvNet, self).__init__()
		InvNetModel = importlib.import_module('model.' + model).__getattribute__(model)
		LossFc = importlib.import_module('lossfc.' + lossfc).__getattribute__(lossfc)
		
		self.__S__ = InvNetModel().cuda()
		self.__L__ = LossFc().cuda()
		self.__optimizer__ = torch.optim.Adam(self.parameters(), lr = lr)

	## ===== ===== ===== ===== ===== ===== ===== =====
	## Train network
	## ===== ===== ===== ===== ===== ===== ===== =====
	def train_network(self, it, learningrate, loader, **kwargs):
		self.train()

		stepsize = loader.batch_size
		loss = 0

		for i, (data, mask) in enumerate(loader):

			self.zero_grad()
			out = self.__S__.forward(data.cuda())
			nloss = self.__L__.forward(out, mask)
			nloss.backward()

			loss += nloss.detach().cpu().numpy()

			self.__optimizer__.step()

		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + \
		" Training %d it, learningrate = %f."%(it, learningrate) + \
		" Processing (%d/%d): (%.3f%%) "%(i + 1, loader.__len__(), 100 * ((i+1) / loader.__len__())) + \
		" Loss %f \r"%(loss / (i + 1)))
		sys.stderr.flush()
		sys.stderr.write("\n")
		return loss

	def validate_network(self, it, loader, **kwargs):
		self.eval()

		stepsize = loader.batch_size
		loss = 0

		for i, (data, mask) in enumerate(loader):
			with torch.no_grad():
				out = self.__S__.forward(data.cuda())
				nloss = self.__L__.forward(out, mask)
				loss += nloss.detach().cpu().numpy()

		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + \
		" Validation %d it"%(it) + \
		" Processing (%d/%d): (%.3f%%) "%(i + 1, loader.__len__(), 100 * ((i+1) / loader.__len__())) + \
		" Loss %f \r"%(loss / (i+1)))
		sys.stderr.flush()
		sys.stderr.write("\n")
		return loss

	def visualization(self, data, path, **kwargs):
		sys.stdout.write('\n' + time.strftime("%Y-%m-%d %H:%M:%S") + ' visualization_begin. \n')
		sys.stderr.flush()
		self.eval()

		data_list = glob.glob(data + "/*")
		for data_name in data_list:
			image = sio.loadmat(data_name)['clean_data']
			min_d = -9.0
			max_d = 9.0
			image1 = (image-min_d)/(max_d-min_d)
			data = torch.FloatTensor(image1).unsqueeze(0)
			data = torch.unsqueeze(data, 0)
			with torch.no_grad():
				out = self.__S__.forward(data.cuda()).detach().cpu().numpy()[0][0]
				sio.savemat(path + '/' + (data_name.split('.')[-2]).split('/')[-1]+'.mat', {'pred': out})
		
		sys.stdout.write('\n' + time.strftime("%Y-%m-%d %H:%M:%S") + ' visualization_end. \n')
		sys.stderr.flush()

	## ===== ===== ===== ===== ===== ===== ===== =====
	## Update learning rate
	## ===== ===== ===== ===== ===== ===== ===== =====
	def updateLearningRate(self, alpha):

		learning_rate = []
		for param_group in self.__optimizer__.param_groups:
			param_group['lr'] = param_group['lr'] * alpha
			learning_rate.append(param_group['lr'])

		return learning_rate

	## ===== ===== ===== ===== ===== ===== ===== =====
	## Save the model
	## ===== ===== ===== ===== ===== ===== ===== =====
	def saveParameters(self, path):

		torch.save(self.state_dict(), path, _use_new_zipfile_serialization=False)

	## ===== ===== ===== ===== ===== ===== ===== =====
	## Load the model's paramters
	## ===== ===== ===== ===== ===== ===== ===== =====
	def loadParameters(self, path):

		self.load_state_dict(torch.load(path))
		# self_state = self.state_dict()
		# loaded_state = torch.load(path)
		# for name, param in loaded_state.items():
		# 	origname = name;
		# 	if name not in self_state:
		# 		name = name.replace("module.", "")

		# 		if name not in self_state:
		# 			# print("%s is not in the model."%origname)
		# 			continue

		# 	if self_state[name].size() != loaded_state[origname].size():
		# 		sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
		# 		continue

		# 	self_state[name].copy_(param)
