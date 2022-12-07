#工具类
import torch
import pandas as pd
import numpy as np
class utils():
	def __init__(self) -> None:
		super().__init__()
	def loadKinase(self):
		data = pd.read_csv("./data/kinase_sequence.csv").values
		data = torch.from_numpy(data.astype(np.float64))
		return data
	def loadCompound(self):
		data = pd.read_csv("./data/compound_feature166.csv").values
		data = torch.from_numpy(data.astype(np.float64))
		return data


	def loadMetaLongTrain(self):
		data = pd.read_csv("./data/metaData/metaLongTrain.csv").values
		data = torch.from_numpy(data.astype(np.int))
		return data
	def loadMetaTailTrain(self):
		data = pd.read_csv("./data/metaData/metaTailTrain.csv").values
		data = torch.from_numpy(data.astype(np.int))
		return data
	def loadMetaLongTest(self,):
		data = pd.read_csv("./data/metaData/metaLongTest.csv").values
		data = torch.from_numpy(data.astype(np.int))
		return data
	def loadMetaTailTest(self):
		data = pd.read_csv("./data/metaData/metaTailTest.csv").values
		data = torch.from_numpy(data.astype(np.int))
		return data
if __name__ == '__main__':
    u = utils()
    print(u.loadMetaLongTrain())

