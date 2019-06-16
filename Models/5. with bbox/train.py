import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.backends.cudnn.benchmark = True
from data_loader import cars_train, cars_val

import copy #for copying
import pickle

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

num_epochs = 3000
batch_size = 64
lr = 0.001

#data loading
train_data = cars_train() #for the  testing part
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 4)
train_data_len = len(train_data)

test_data = cars_val()
test_loader = DataLoader(test_data, batch_size = batch_size, num_workers = 4)
test_data_len = len(test_data)

#model initialization and multi GPU
device = torch.device ("cuda:0" if torch.cuda.is_available() else "cpu")

#resnet not pre-trained one
model = models.resnet50(pretrained = True)
model.fc = nn.Linear(2048, 196) #for the classification

#using more than 1 GPU if available
if (torch.cuda.device_count() > 1):
	model = nn.DataParallel(model)
model = model.to(device)

#loss, optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 5e-5)
scheduler = ReduceLROnPlateau(optimizer, mode = "max", patience = 10, min_lr = 1e-5)

print ("learning started ")
best_acc1 = 0.0
best_wts = None
for epoch in range(num_epochs):
	train_acc1 = 0.0
	epoch_loss = 0.0

	model.train()
	for i, (inputs,labels) in enumerate(train_loader):


		inputs = inputs.to(device) #change to device
		labels = labels.to(device, dtype = torch.long)

		predictions = model(inputs) #label since output_mode is just error

		#now calculate the loss function
		loss = criterion(predictions, labels)


		#backprop here
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		epoch_loss += (loss.data * inputs.shape[0])
		acc1, = accuracy(predictions, labels)
		train_acc1 += (acc1 )
		#print('Ep_Tr:{}/{},step:{}/{},top1:{},loss:{}'.format(epoch, num_epochs, i, train_data_len //batch_size,train_acc1, loss.data))
	epoch_loss = (epoch_loss / float(train_data_len))
	print('Ep_Tr: {}/{}, top1:{}, ls:{}'.format(epoch, num_epochs,train_acc1/float(train_data_len),epoch_loss.data))


	#validation
	model.eval()
	val_acc1 = 0.0
	for i, (inputs,labels) in enumerate(test_loader):

		inputs = inputs.to(device) #change to device
		labels = labels.to(device, dtype = torch.long)

		with torch.no_grad():
			predictions = model(inputs)

		acc1, = accuracy(predictions, labels)
		val_acc1 += (acc1 )
		#print('Ep_vl: {}/{},step: {}/{},top1:{}'.format(epoch, num_epochs, i, test_data_len //batch_size,val_acc1))

	val_acc1 = val_acc1 / float(test_data_len)
	print('Ep_vl: {}/{}, top1:{}'.format(epoch, num_epochs,val_acc1))

	scheduler.step(val_acc1) #for the scheduler

	if (best_acc1 < val_acc1):
		best_acc1 = val_acc1
		best_model_wts = copy.deepcopy(model.state_dict())
		torch.save(best_model_wts, 'resnet50_adam_weights.pt')
	print('Epoch: {}/{}, top1: {}'.format(epoch, num_epochs, best_acc1)) #print the epoch loss

torch.save(best_model_wts, 'resnet50_adam_weights.pt') #save the best model weights

f = open("best_acc_resnet50_adam", 'w')
f.write(str(best_acc1) + "\n")
f.close()
