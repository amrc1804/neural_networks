#unet paper: https://arxiv.org/pdf/1505.04597
#pytorch doc: https://docs.pytorch.org/docs/2.14/torch.html

import torch
import torch.nn as nn
import torch.nn.functionas F
import numpy as np

class unet(nn.Module):

	def __init__(self):
		super().__init__()
		
		#Contraction step, "descent" of the U shape
		#parameters of the first Conv2d method:
			#in_channels = 1, number of channels of the image, here grayscale so 1 but if rbg it's 3
			#out_channels = 64, number of output channels after the convolution, 64 as in the paper, so 64 kernels
			#kernel_size = 3, 3x3 square (see paper)
		self.conv1 = nn.Conv2d(1, 64, 3)
		self.conv2 = nn.Conv2d(64, 64, 3)
		#first max pool
		self.conv3 = nn.Conv2d(64, 128, 3)
		self.conv4 = nn.Conv2d(128, 128, 3)
		#second max pool
		self.conv5 = nn.Conv2d(128, 256, 3)
		self.conv6 = nn.Conv2d(256, 256, 3)
		#third max pool
		self.conv7 = nn.Conv2d(256, 512, 3)
		self.conv8 = nn.Conv2d(512, 512, 3)
		#fourth and last max pool
		self.conv9 = nn.Conv2d(512, 1024, 3)
		self.conv10 = nn.Conv2d(1024, 1024, 3)
		
		#Expansion step, "ascent" of the U shape
		#parameters of the first ConvTranspose2d method:
			#in_channels = 1024, number of channels of the last convolution from the descending step
			#out_channels = 512, in the paper those channels will be concatenated with the 512 channels from the mirrored step of the descent
			#kernel_size = 2, from the paper
			#stride = 2, upsampling of the image, size is 2x what is was before
		self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
		self.conv11 = nn.Conv2d(1024, 512, 3)
		self.conv12 = nn.Conv2d(512, 512, 3)
		
		self.upconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
		self.conv13 = nn.Conv2d(512, 256, 3)
		self.conv14 = nn.Conv2d(256, 256, 3)
		
		self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
		self.conv15 = nn.Conv2d(256, 128, 3)
		self.conv16 = nn.Conv2d(128, 128, 3)
		
		self.upconv4 = nn.ConvTranspose2d(128, 64, 2, 2)
		self.conv17 = nn.Conv2d(128, 64, 3)
		self.conv18 = nn.Conv2d(64, 64, 3)
		
		#last conv, linear connection pretty much?
		self.conv19 = nnConv2d(64, 2, 1)
		
	def forward(self, input):
		
		#Followed the naming from the pytorch doc, c is convolution, s subsampling, u is up-conv etc...
		c1 = F.relu(self.conv1(input))
		c2 = F.relu(self.conv2(c1))
		s1 = F.max_pool2d(c1, (2, 2))
		
		c3 = F.relu(self.conv3(s1))
		c4 = F.relu(self.conv4(c3))
		s2 = F.max_pool2d(c4, (2, 2))
		
		c5 = F.relu(self.conv5(s2))
		c6 = F.relu(self.conv6(c5))
		s3 = F.max_pool2d(c6, (2, 2))
		
		c7 = F.relu(self.conv7(s3))
		c8 = F.relu(self.conv8(c7))
		s4 = F.max_pool2d(c8, (2, 2))
	
		c9 = F.relu(self.conv9(s4))
		c10 = F.relu(self.conv10(c9))
		u1 = self.upconv1(c10)
		
		#concatenation of previous step of the descent with output of the upconv, core of the unet
		cc1 = torch.cat(c8, u1, 1)
		c11 = F.relu(self.conv11(cc1))
		c12 = F.relu(self.conv12(c11))
		u2 = self.upconv2(c12)
		
		cc2 = torch.cat(c6, u2, 1)
		c13 = F.relu(self.conv13(cc2))
		c14 = F.relu(self.conv14(c13))
		u3 = self.upconv3(c14)
		
		cc3 = torch.cat(c4, u3, 1)
		c15 = F.relu(self.conv15(cc3))
		c16 = F.relu(self.conv16(c15))
		u4 = self.upconv4(c16)
		
		cc4 = torch.cat(c2, u4, 1)
		c17 = F.relu(self.conv15(cc4))
		c18 = F.relu(self.conv16(c17))
		
		output = self.conv19(c18)
		return output
		