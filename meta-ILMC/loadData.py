from utils import utils
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
Q_LENGTH = 64
class myDataset(Dataset):
	def __init__(self, train = "longTrain", task = 1):
		UTILS = utils()
		self.kinaseEmbedding = UTILS.loadKinase()
		self.compoundEmbedding = UTILS.loadCompound()
		if train == "longTrain":
			Y = UTILS.loadMetaLongTrain()
			Y = Y[Y[:,1] == task]
		elif train == "longTestOne":
			Y = UTILS.loadMetaLongTest()
			Y = Y[Y[:, 1] == task]
			Y_ONE = np.array(Y[Y[:,2] == 1])
			Y = torch.from_numpy(Y_ONE)
		elif train == "longTestZero":
			Y = UTILS.loadMetaLongTest()
			Y = Y[Y[:, 1] == task]
			Y_ZERO = np.array(Y[Y[:, 2] == 0])
			Y = torch.from_numpy(Y_ZERO)
		elif train == "tailTrain":
			Y = UTILS.loadMetaTailTrain()
			Y = Y[Y[:, 1] == task]
		elif train == "tailTest":
			Y = UTILS.loadMetaTailTest()
			Y = Y[Y[:, 1] == task]
		self._x = Y[:,:2]
		self._y = Y[:,2]
		self._len = len(Y)
	def __len__(self) -> int:
		return self._len
	def __getitem__(self, index: int):
		kinaseIndex = self._x[index][1]
		compoundIndex = self._x[index][0]
		kinase = self.kinaseEmbedding[kinaseIndex]
		compound = self.compoundEmbedding[compoundIndex]
		return compound, kinase, self._y[index]
if __name__ == '__main__':
	data0 = DataLoader(myDataset(dataPath = "10_1000_1", train = "longTestZero", task = 1),
	                  batch_size=32, shuffle=True, drop_last=False, num_workers=0)
	data1 = DataLoader(myDataset(dataPath = "10_1000_1", train = "longTestOne", task = 1),
	                  batch_size=32, shuffle=True, drop_last=False, num_workers=0)
	for i, data in enumerate(zip(data0, data1)):
		print(torch.cat((data[0][0], data[1][0]), 0))
		break