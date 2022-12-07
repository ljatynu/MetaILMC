from model import IMC
from utils import utils
from torch.utils.data import Dataset, DataLoader, random_split
import torch
UTILS = utils()
class myDataset(Dataset):
	def __init__(self, train = True, fold = 0):
		self.kinaseEmbedding = UTILS.loadKinase("./data/")
		self.compoundEmbedding = UTILS.loadCompound("./data/")
		if train:
			Y = UTILS.loadTrain("./data/tenFold/" + str(fold) + "/")
		else:
			Y = UTILS.loadTest("./data/tenFold/" + str(fold) + "/")

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