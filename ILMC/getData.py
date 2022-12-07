from model import IMC
from utils import utils
from torch.utils.data import Dataset, DataLoader, random_split
import torch
DATASETPATH = "./data/merged/"
UTILS = utils()
class myDataset(Dataset):
	def __init__(self, DATASETPATH):
		self.kinaseEmbedding = UTILS.loadKinase(DATASETPATH)
		self.compoundEmbedding = UTILS.loadCompound(DATASETPATH)
		Y = UTILS.loadY(DATASETPATH)

		self._x = Y[:,:2]
		self._y = Y[:,2]
		self._len = len(Y)
	def __getitem__(self, index: int):
		kinaseIndex = self._x[index][1]
		compoundIndex = self._x[index][0]
		kinase = self.kinaseEmbedding[kinaseIndex]
		compound = self.compoundEmbedding[compoundIndex]
		return  compound, kinase, self._y[index]
	def __len__(self):
		return self._len
if __name__ == '__main__':
	data = myDataset(DATASETPATH)
	train_size, test_size = 145958, data._len - 145958
	train_set, test_set = random_split(data, [train_size, test_size])
	train_data = DataLoader(train_set, batch_size=10, shuffle=True, drop_last=False, num_workers=0)
	test_data = DataLoader(test_set, batch_size=10, shuffle=True, drop_last=False, num_workers=0)
	print(len(train_data))
	print(len(test_data))
	model = IMC(dimKinase = 343, dimCompound = 167, dimHidden = 64)
	model.train()
	for compound, kinase, y in train_data:
		predY = model(kinase.type(torch.float), compound.type(torch.float))
		print(predY)
		break
