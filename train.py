import unet
import dataloader_flyingchair as dlfl
import time
import torch
import torch.nn as nn
from torch.backends import cudnn
import torchvision.transforms as transforms
import matplotlib
#apparently so I don't have error on the cluster
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def epoch(model, data, criterion, print_interval, optimizer=None, cuda=False):

	is_train = optimizer is not None
	model.train() if is_train else model.eval()

	avg_loss = AverageMeter()
	avg_batch_time = AverageMeter()

	tic = time.time()
	with torch.set_grad_enabled(is_train):
		for i, (inpt, target) in enumerate(data):

			if cuda:
				inpt = inpt.cuda()
				target = target.cuda()

			output = model(inpt)
			loss = criterion(output, target)

			if is_train:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			batch_time = time.time() - tic
			tic = time.time()

			avg_loss.update(loss.item())
			avg_batch_time.update(batch_time)

			if i % print_interval == 0:
				print(f"{'TRAIN' if optimizer else 'EVAL'} Batch {i}/{len(data)}\t", f"Time {avg_batch_time.val:.3f}s ({avg_batch_time.avg:.3f}s)\t", f"Loss {avg_loss.val:.4f} ({avg_loss.avg:.4f})\t")

	print(f"\n Total time {int(avg_batch_time.sum):d}s\t", f"Avg loss {avg_loss.avg:.4f}\t")
	return avg_loss

def prediction(model, nb_pred, test_batches, cuda=True):

	model.eval()
	saved = 0
	if nb_pred == -1:
		nb_pred = len(test_batches)
	print(nb_pred)
	maes = []

	std_t = torch.tensor([0.27410381, 0.28123705, 0.31550989], dtype=torch.float32).view(3,1,1)
	mean_t = torch.tensor([0.3804277, 0.38884523, 0.3820099], dtype=torch.float32).view(3,1,1)
	if cuda:
		std_t = std_t.cuda()
		mean_t = mean_t.cuda()

	with torch.no_grad():
		for inpt, label in test_batches:
			if cuda:
				inpt = inpt.cuda()
				label = label.cuda()
			output = model(inpt)
			mse = F.mse_loss(output, label)
			batch_size = output.shape[0]

			for b in range(batch_size):
				if saved >= nb_pred:
					break 
				image2pred = output[b] * std_t + mean_t
				label2 = label[b] * std_t + mean_t
				#print(type(image2pred), type(label2))
				label_np = torch.clamp(label2.permute(1, 2, 0), 0.0, 1.0).detach().cpu().numpy()
				output_np = torch.clamp(image2pred.permute(1, 2, 0), 0.0, 1.0).detach().cpu().numpy()
				plt.imsave(f"preds/image2label_{saved}.png", label_np)
				plt.imsave(f"preds/image2pred_{saved}.png", output_np)
				mae_map = F.l1_loss(image2pred, label2, reduction='none')
				mae = mae_map.mean().item()
				mae_map =  torch.clamp(mae_map.permute(1, 2, 0), 0.0, 1.0).detach().cpu().numpy()
				plt.imsave(f"preds/image2mae_{saved}.png", mae_map)
				maes.append(mae)
				saved += 1

	return np.array(maes), np.array(maes).mean()

def main(data_dir, split_path, batch_size=128, lr=0.001, epochs=10, print_interval=50, cuda=False, train=False, modelt=None):

	model = unet.Unet()
	if modelt:
		model.load_state_dict(torch.load('unet_advection.pth')) 
	
	if train:
		criterion = nn.MSELoss()
		#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	if cuda:
		cudnn.benchmark = True
		model = model.cuda()
		if train:
			criterion = criterion.cuda()

	dataset = dlfl.CustomImageDataset(data_dir, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.3804277 , 0.38884523, 0.3820099), (0.27410381, 0.28123705, 0.31550989))]))
	train_batches, test_batches, val_batches = dlfl.load_data(dataset, split_path, batch_size)

	train_losses = []
	test_losses = []

	if train:
		for i in range(epochs):
			print("=================\n=== EPOCH "+str(i+1)+" =====\n=================\n")
			train_loss = epoch(model, train_batches, criterion, print_interval, optimizer=optimizer, cuda=cuda)
			test_loss = epoch(model, test_batches, criterion, print_interval, cuda=cuda)

			train_losses.append(train_loss.avg)
			test_losses.append(test_loss.avg)

			plt.figure()
			plt.plot(train_losses, label='Train Loss')
			plt.plot(test_losses, label='Test Loss')
			plt.xlabel('Epoch')
			plt.ylabel('Loss')
			plt.legend()
			plt.savefig('preds/loss_curve.png')
			plt.close()
		torch.save(model.state_dict(), 'unet_advection.pth')
	maes, mae = prediction(model, -1, val_batches)
	print(f"MAE: {mae}")
	#plt.figure()
	#plt.bar(len(maes), maes, label=f"Average MAE: {mae}")
	#plt.xlabel('Image')
	#plt.ylabel('MAE')
	#plt.savefig('mae.png')
	#plt.close()

if __name__ == '__main__':
	main(
		data_dir = '',
		split_path = 'FlyingChairs_train_val.txt',
		batch_size = 32,
		lr = 0.001,
		epochs = 50,
		print_interval = 50,
		cuda=True,
		train=True
		#train=False,
		#modelt='unet_advection.pth'
	)
