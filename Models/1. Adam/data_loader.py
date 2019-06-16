import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image
import scipy.io

class cars_train(Dataset):
	"""
	Dataset class for Cars Training part.
	"""
	def __init__(self):

		self.all = scipy.io.loadmat("cars_train_annos.mat")
		self.files = []
		self.labels = []

		for i in range(len(self.all["annotations"][0])):
			self.files.append("./cars_train/" + self.all["annotations"][0][i][-1][0] )
			self.labels.append(self.all["annotations"][0][i][-2][0][0] - 1.0)

		self.length = len(self.files)
		self.transform = transforms.Compose([transforms.Resize((320)),transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(), transforms.ToTensor(),
						  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	def __getitem__(self, index):

		fl = self.files[index] #get the proper file
		category = float(self.labels[index])

		image = self.transform( Image.open( fl ).convert("RGB") )

		return (image, category )

	def __len__(self):
		return self.length

class cars_val(Dataset):
	"""
	Dataset class for cars validation part.
	"""
	def __init__(self):

		self.all = scipy.io.loadmat("cars_test_annos.mat")
		self.files = []
		self.labels = []

		for i in range(len(self.all["annotations"][0])):
			self.files.append("./cars_test/" + self.all["annotations"][0][i][-1][0] )
			self.labels.append(self.all["annotations"][0][i][-2][0][0] - 1.0)

		self.length = len(self.files)
		self.transform = transforms.Compose([transforms.Resize((320)),transforms.CenterCrop(224),transforms.ToTensor(),
						  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	def __getitem__(self, index):

		fl = self.files[index] #get the proper file
		category = float(self.labels[index])

		image = self.transform( Image.open( fl ).convert("RGB") )

		return (image, category )

	def __len__(self):
		return self.length
