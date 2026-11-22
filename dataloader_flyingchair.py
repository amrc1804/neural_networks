import numpy as np
#from scipy.misc import imread
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split
import pandas as pd
import glob
import os 

# objectives: 1. open the images (.ppm for image, .flo for gt) 2. update, task 1 label is image2 not flow

#Functions to read a .flo image taken from flying chairs utils: https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py

def readFlow(name):
	if name.endswith('.pfm') or name.endswith('.PFM'):
		return readPFM(name)[0][:,:,0:2]

	f = open(name, 'rb')

	header = f.read(4)
	if header.decode("utf-8") != 'PIEH':
		raise Exception('Flow file header does not contain PIEH')

	width = np.fromfile(f, np.int32, 1).squeeze()
	height = np.fromfile(f, np.int32, 1).squeeze()

	flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

	return flow.astype(np.float32)

def writeFlow(name, flow):
	f = open(name, 'wb')
	f.write('PIEH'.encode('utf-8'))
	np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
	flow = flow.astype(np.float32)
	flow.tofile(f)


def readImagePPM(name):

	#I discarded all the code about the .pfm as it's irrelevant for flyingchairs, may put it back if it turns out to be needed
	return np.array(Image.open(name)) #imread(name)

#Make the dataloader, data is like that: 5 digit number from 00001 to 22872 followed by either _flow.flo for the flowmap, or _img1/2.ppm for the images
#So my idea is to concatenate the two images (flownet)

#taken from pytorch doc: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomImageDataset(Dataset):

	def __init__(self, data_dir, transform=None, target_transform=None):
		self.img_dir = data_dir
		self.transform = transform
		self.target_transform = target_transform

		self.img2_paths = sorted(glob.glob(os.path.join(data_dir, "*_img2.ppm")))

	def __len__(self):
		return len(self.img2_paths)

	def __getitem__(self, idx):

		base_name = os.path.basename(self.img2_paths[idx])
		file_id = base_name.split('_')[0]

		img1_path = os.path.join(self.img_dir, f"{file_id}_img1.ppm")
		img2_path = os.path.join(self.img_dir, f"{file_id}_img2.ppm")
		flow_path = os.path.join(self.img_dir, f"{file_id}_flow.flo")

		img1 = readImagePPM(img1_path)
		img2 = readImagePPM(img2_path)
		flow = readFlow(flow_path)

		if self.transform:
			img1 = self.transform(img1)
			img2 = self.transform(img2)
		if self.target_transform:
			flow = self.target_transform(flow)
		else:
			flow = torch.from_numpy(np.transpose(flow, (2, 0, 1)))

		x = torch.cat((img1, flow), dim=0)

		return x, img2

def load_data(dataset, split_data_path, batch_size, num_workers=4, pin_memory=True):
	split_values = np.loadtxt(split_data_path, dtype=int)

	train_indices = np.where(split_values == 1)[0].tolist()
	test_indices = np.where(split_values == 2)[0].tolist()

	train_split = Subset(dataset, train_indices)
	test_val_split = Subset(dataset, test_indices)
	test_split, val_split  = random_split(test_val_split, [0.5, 0.5])

	persistent = num_workers > 0

	train_batches = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
	test_batches = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
	val_batches = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)

	return train_batches, test_batches, val_batches


def compute_mean_std(data_dir):
	img_paths = glob.glob(os.path.join(data_dir, "*.ppm"))

	pixel_sum = np.zeros(3, dtype=np.float64)
	pixel_sum_sq = np.zeros(3, dtype=np.float64)
	num_pixels = 0

	for path in img_paths:
		img = readImagePPM(path) / 255.0
		pixel_sum += np.sum(img, axis=(0, 1))
		pixel_sum_sq += np.sum(img**2, axis=(0, 1))
		num_pixels += img.shape[0] * img.shape[1]

	mean = pixel_sum / num_pixels
	std = np.sqrt((pixel_sum_sq / num_pixels) - (mean**2))

	return mean, std

#(array([0.3804277 , 0.38884523, 0.3820099 ]), array([0.27410381, 0.28123705, 0.31550989]))
#data_dir = '/users/Enseignants/bereziat/FlyingChairs/data'
#print(compute_mean_std(data_dir))
