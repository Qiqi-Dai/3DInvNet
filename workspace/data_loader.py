import os, torch, numpy, glob
import matplotlib.image as mpimg
import scipy.io as sio

## ===== ===== ===== ===== ===== ===== ===== =====
## Load the data
## ===== ===== ===== ===== ===== ===== ===== =====
def load_data(data_name):
	image = sio.loadmat(data_name)['clean_data']
	min_d = -9.0
	max_d = 9.0
	image1 = (image-min_d)/(max_d-min_d)
	data = torch.FloatTensor(image1)
	data1 = torch.unsqueeze(data, 0)
	return data1

def load_mask(mask_name):
	image = sio.loadmat(mask_name)['mask']
	mask = torch.FloatTensor(image)
	mask1 = torch.unsqueeze(mask, 0)
	return mask1

class DatasetLoader(object):
	def __init__(self, data, mask, **kwargs):
		self.data_path = data
		self.mask_path = mask
		self.data = []
		self.mask = []
		data_list = glob.glob(self.data_path + "/*")
		mask_list = glob.glob(self.mask_path + "/*")
		for data_name in data_list:
			index = (data_name.split('.')[-2]).split('/')[-1]
			data_name = self.data_path + "/%s.mat"%(index)
			mask_name = self.mask_path + "/%s.mat"%(index)
			self.data.append(data_name)
			self.mask.append(mask_name)

	def __getitem__(self, index):
		data = load_data(self.data[index])
		mask = load_mask(self.mask[index])
		return data, mask

	def __len__(self):
		return len(self.data)