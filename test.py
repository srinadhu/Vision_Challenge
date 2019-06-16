import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
torch.backends.cudnn.benchmark = True

from PIL import Image
import scipy.io

import copy #for copying
import pickle
import csv

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k)
	return res



class cars_test(Dataset):
	"""
	Dataset class for cars test part.
	"""
	def __init__(self):

		self.all = scipy.io.loadmat("cars_test_annos.mat") #change this accordingly
		self.files = []
		self.labels = []

		self.bbox_x1 = []
		self.bbox_y1 = []
		self.bbox_x2 = []
		self.bbox_y2 = []

		for i in range(len(self.all["annotations"][0])):

			self.files.append("./cars_test/" + self.all["annotations"][0][i][-1][0] ) #this as well
			self.labels.append(self.all["annotations"][0][i][-2][0][0] - 1.0)

			self.bbox_x1.append(self.all["annotations"][0][i][0][0][0])
			self.bbox_y1.append(self.all["annotations"][0][i][1][0][0])
			self.bbox_x2.append(self.all["annotations"][0][i][2][0][0])
			self.bbox_y2.append(self.all["annotations"][0][i][3][0][0])

		self.length = len(self.files)
		self.transform = transforms.Compose([transforms.Resize((256)),transforms.CenterCrop(224),transforms.ToTensor(),
						  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	def __getitem__(self, index):

		fl = self.files[index] #get the proper file
		category = float(self.labels[index])

		image = self.transform( Image.open( fl ).convert("RGB").crop( ( self.bbox_x1[index], self.bbox_y1[index], self.bbox_x2[index], self.bbox_y2[index] ) ) )

		return (image, category, fl)

	def __len__(self):
		return self.length


#data loading
test_data = cars_test()
test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 4)
test_data_len = len(test_data)

#model initialization and multi GPU
device = torch.device ("cuda:0" if torch.cuda.is_available() else "cpu")

#resnet not pre-trained one
model = models.resnet50(pretrained = True)
model.fc = nn.Linear(2048, 196) #for the classification

model.load_state_dict(torch.load("resnet50_adam_weights.pt"))

#using more than 1 GPU if available
if (torch.cuda.device_count() > 1):
	model = nn.DataParallel(model)
model = model.to(device)

print ("testing started ")

#testing
model.eval()
acc = 0.0
for i, (inputs,labels, files) in enumerate(test_loader):

	inputs = inputs.to(device) #change to device
	labels = labels.to(device, dtype = torch.long)

	with torch.no_grad():
		predictions = model(inputs)

	acc1, = accuracy(predictions, labels)
	acc += (acc1 )
		
acc = acc / float(test_data_len)
print('Top1 Accuracy:{}'.format(acc))

