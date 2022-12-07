#工具类
import pandas as pd
import numpy as np
import torch
class utils():
	def __init__(self) -> None:
		super().__init__()
	#读取激酶，化合物的Embedding
	def loadKinase(self, datasetPath = "./data/"):
		data = pd.read_csv(datasetPath + "kinase_sequence.csv").values
		data = torch.from_numpy(data.astype(np.float64))
		return data
	def loadCompound(self, datasetPath = "./data/"):
		data = pd.read_csv(datasetPath + "compound_feature166.csv").values
		data = torch.from_numpy(data.astype(np.float64))
		return data
	#十折实验读取数据
	def loadTrain(self, datasetPath = "./data/"):
		data = pd.read_csv(datasetPath + "train.csv").values
		data = torch.from_numpy(data.astype(np.int))
		return data
	def loadTest(self, datasetPath = "./data/"):
		data = pd.read_csv(datasetPath + "test.csv").values
		data = torch.from_numpy(data.astype(np.int))
		return data

	#长尾实验读取数据
	def loadLongTailTrain(self, datasetPath = "./data/longTail/"):
		data = pd.read_csv(datasetPath + "longTailTrain.csv").values
		data = torch.from_numpy(data.astype(np.int))
		return data
	def loadLongTest(self, datasetPath = "./data/longTail/"):
		data = pd.read_csv(datasetPath + "longTest.csv").values
		data = torch.from_numpy(data.astype(np.int))
		return data
	def loadTailTest(self, datasetPath = "./data/longTail/"):
		data = pd.read_csv(datasetPath + "tailTest.csv").values
		data = torch.from_numpy(data.astype(np.int))
		return data
if __name__ == '__main__':
    u = utils()
    u.loadY("./data/")