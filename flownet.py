#flownet paper: https://openaccess.thecvf.com/content_cvpr_2017/papers/Ilg_FlowNet_2.0_Evolution_CVPR_2017_paper.pdf
#pytorch doc: https://docs.pytorch.org/docs/2.14/torch.html

import torch
from torch import nn
import torch.nn.functionnal as F

class flownet(nn.Module):
	def __init__(self):
		super().__init__()
		
		#This architecture doesn't use max/average pooling, but use a stride of 2 at each convolution, first convolution has a 7x7 kernel
		#Remark, it may be a good idea to swap the stride for a max pooling to check for possible improvements
		#input channels is 6, because we concatenate the two rgb images
		self.conv1 = nn.Conv2d(6, 64, 7, padding=3, stride=2)
		self.conv2 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
		self.conv3 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
		self.conv3_1 = nn.Conv2d(256, 256, 3, padding=1)
		self.conv4 = nn.Conv2d(256, 512, 3, padding=1, stride=2)
		self.conv4_1 = nn.Conv2d(512, 512, 3, padding=1)
		self.conv5 = nn.Conv2d(512, 512, 3, padding=1, stride=2)
		self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
		self.conv6 = nn.Conv2d(512, 1024, 3, padding=1, stride=2)
		
		#Refinement
		self.deconv5 = nn.ConvTranspose2d(1024, 512, 5, 2, 2, 1)
		self.flow5 = nn.Conv2d(1024, 2, 5, padding=2)
		self.deconv4 = nn.ConvTranspose2d(1024, 256, 5, 2, 2, 1)
		self.flow4 = nn.Conv2d(770, 2, 5, padding=2)
		#From by understanding, we concatenate deconv4 and conv4_1 and we also add flow5 at the end, which result in 513 input channels
		self.deconv3 = nn.ConvTranspose2d(770, 128, 5, 2, 2, 1)
		self.flow3 = nn.Conv2d(386, 2, 5, padding=2)
		self.deconv2 = nn.ConvTranspose2d(386, 64, 5, 2, 2, 1)
		self.prediction = nn.Conv2d(194, 2, 5, padding=2)
		
	def forward(self, x):
		
		conv1 = self.conv1(x)
		conv1 = F.relu(conv1)
		conv2 = self.conv2(conv1)
		conv2 = F.relu(conv2)
		conv3 = self.conv3(conv2)
		conv3 = F.relu(conv3)
		conv3_1 = self.conv3_1(conv3)
		conv3_1 = F.relu(conv3_1)
		conv4 = self.conv4(conv3_1)
		conv4 = F.relu(conv4)
		conv4_1 = self.conv4_1(conv4)
		conv4_1 = F.relu(conv4_1)
		conv5 = self.conv5(conv4_1)
		conv5 = F.relu(conv5)
		conv5_1 = self.conv5_1(conv5)
		conv5_1 = F.relu(conv5_1)
		conv6 = self.conv6(conv5_1)
		conv6 = F.relu(conv6)
		
		#refinement
		deconv5 = self.deconv5(conv6)
		#dim=1 because the input is of shape N * C * H * W with N the batch size, C the number of channels, H the height of the image and W its width
		cat1 = torch.cat((deconv5, conv5_1), dim=1)
		flow5 = self.flow5(cat1)
		flow5 = F.upsample(flow5, scale_factor=2)
		
		deconv4 = self.deconv4(cat1)
		cat2 = torch.cat((deconv4, conv4_1, flow5), dim=1)
		flow4 = self.flow4(cat2)
		flow4 = F.upsample(flow4, scale_factor=2)
		
		deconv3 = self.deconv3(cat2)
		cat3 = torch.cat((deconv3, conv3_1, flow4), dim=1)
		flow3 = self.flow3(cat3)
		flow3 = F.upsample(flow3, scale_factor=2)
		
		deconv2 = self.deconv2(cat3)
		cat4 = torch.cat((deconv2, conv2_1, flow3), dim=1)
		out = self.prediction(cat)
		
		return out