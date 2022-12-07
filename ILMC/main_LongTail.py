from torch.utils.data import DataLoader, random_split
from getData import myDataset
from model import IMC
from torch import nn, optim
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import torch.functional as F
import longTailData
import computeKinaseAUC
from Metrics import *
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 5000
LEARNING_RATE = 1e-3
EPOCHS = 1000
DK = 343 #激酶向量的维度
DC = 167 #化合物向量的维度
DH = 64 #隐藏层的维度
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(compound, kinase, y, model, optimizer):
	model.train()
	optimizer.zero_grad() #梯度置为0
	predY = model(kinase.type(torch.float), compound.type(torch.float)) #进行预测
	lossFunction = nn.BCELoss().to(device)
	loss = lossFunction(torch.squeeze(predY), torch.squeeze(y.type(torch.float)))  # 计算损失
	loss.backward() #反向传播
	optimizer.step() #更新权重
	return loss.item()
def test(compound, kinase, y, model):
	with torch.no_grad():
		predY = model(kinase.type(torch.float), compound.type(torch.float))
	predY = torch.squeeze(predY)
	lossFunction = nn.BCELoss().to(device)  # 自定义的损失函数
	loss = lossFunction(torch.squeeze(predY), torch.squeeze(y.type(torch.float)))  # 计算损失
	y_hat = np.array(predY.to("cpu"))

	predY[predY >= 0.5] = 1
	predY[predY < 0.5] = 0

	TP = sum(y[predY == 1])
	FP = len(y[predY == 1]) - TP
	FN = sum(y[predY == 0])
	TN = len(y[predY == 0]) - FN
	return int(TP), int(TN), int(FN), int(FP), y_hat, loss.item()

def main():
	#获取数据
	# data = myDataset(DATAPATH)
	# testSize, trainSize = 18245, data._len - 18245
	# trainSet, testSet = random_split(data, [trainSize, testSize])
	dataPath = "./data/longTail/"
	trainSet = longTailData.myDataset(train = 1, longTailPath=dataPath)
	longTestSet = longTailData.myDataset(train = 2, longTailPath=dataPath)
	tailTestSet = longTailData.myDataset(train = 3, longTailPath=dataPath)
	# f = open(dataPath + "result.txt", "a")
	f = open("./result/longTail/EPOCH_result.CSV", "a")
	f.write("LT,epoch, loss, test_loss_all, BA, RECALL, PRECISION, F1, AUPR,AUC, BEST_AUC\n")
	print(dataPath)
	print("======================数据集大小=========================")
	print("训练集大小为：{}".format(len(trainSet)))
	print("头部测试集大小为：{}".format(len(longTestSet)))
	print("尾部测试集大小为：{}".format(len(tailTestSet)))
	print("======================开始训练！=========================")
	trainData = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
	longTestData = DataLoader(longTestSet, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
	tailTestData = DataLoader(tailTestSet, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
	model = IMC(dimKinase = DK, dimCompound = DC, dimHidden = DH).to(device) #获取模型
	# model.reset_parameters()

	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0) #优化器
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
	trainLossList = []
	longLoss, longACC, longAUC, longPrecision, longRecall, longF1 = [], [], [], [], [], []
	tailLoss, tailACC, tailAUC, tailPrecision, tailRecall, tailF1 = [], [], [], [], [], []
	best_auc = 0.
	tailPredY = []
	kinaseAUC, kinaseAUPR, kinasePRECISION, kinaseRECALL, kinaseF1, kinaseBA = computeKinaseAUC.getKinaseID()
	BEST_LONG_AUC = 0
	BEST_TAIL_AUC = 0
	for epoch in range(EPOCHS):
		loss = 0
		for i, (compound, kinase, y) in enumerate(trainData):
			loss += train(compound.to(device), kinase.to(device), torch.unsqueeze(y.to(device), 1), model, optimizer)
		scheduler.step()
		trainLossList.append(loss)
		#头部数据测试
		TP_sum, TN_sum, FN_sum, FP_sum = 0., 0., 0., 0.
		test_loss_all = 0
		y_true, y_hat_all = [], []
		longPredY = []
		for i, (compound, kinase, y) in enumerate(longTestData):
			TP, TN, FN, FP, y_hat, test_loss = test(compound.to(device), kinase.to(device), y.to(device), model)
			test_loss_all += test_loss
			TP_sum += TP
			TN_sum += TN
			FN_sum += FN
			FP_sum += FP
			y_true = y_true + list(np.array(y))
			y_hat_all = y_hat_all + list(y_hat)
		longPredY = y_hat_all
		print("TP:{}  TN:{}  FP:{}  FN:{}".format(TP_sum, TN_sum, FP_sum, FN_sum))

		AUC = evaluationAuroc(y_true, y_hat_all)  # 测试集中只有一个类别时，没有auc
		AUPR = evaluationAUPR(y_true, y_hat_all)
		PRECISION = evaluationPrecision(TP_sum, TN_sum, FP_sum, FN_sum)
		F1 = evaluationF1score(TP_sum, TN_sum, FP_sum, FN_sum)
		RECALL = evaluationRecall(TP_sum, TN_sum, FP_sum, FN_sum)
		BA = evaluationBA(TP_sum, TN_sum, FP_sum, FN_sum)

		if AUC > BEST_LONG_AUC:
			BEST_LONG_AUC = AUC
			torch.save(model, "./result/longTail/Long_IMC.pkl")
		print('头部 Epoch {}: TrainLoss {:.4f} TestLoss {:.4f} BA: {:.4f}  recall:{:.4f} precision:{:.4f} AUPR:{:.4f} F1:{:.4f} AUC:{:.4f} BESTAUC:{}'.format(
				epoch, loss, test_loss_all, BA, RECALL, PRECISION, F1, AUPR,AUC, BEST_LONG_AUC))
		f.write('Long,{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
				epoch, loss, test_loss_all, BA, RECALL, PRECISION, F1, AUPR,AUC, BEST_LONG_AUC))

		#尾部数据测试
		TP_sum, TN_sum, FN_sum, FP_sum = 0., 0., 0., 0.
		test_loss_all = 0
		y_true, y_hat_all = [], []

		for i, (compound, kinase, y) in enumerate(tailTestData):
			TP, TN, FN, FP, y_hat, test_loss = test(compound.to(device), kinase.to(device), y.to(device), model)
			test_loss_all += test_loss
			TP_sum += TP
			TN_sum += TN
			FN_sum += FN
			FP_sum += FP
			y_true = y_true + list(np.array(y))
			y_hat_all = y_hat_all + list(y_hat)

		print("TP:{}  TN:{}  FP:{}  FN:{}".format(TP_sum, TN_sum, FP_sum, FN_sum))

		AUC = evaluationAuroc(y_true, y_hat_all)  # 测试集中只有一个类别时，没有auc
		AUPR = evaluationAUPR(y_true, y_hat_all)
		PRECISION = evaluationPrecision(TP_sum, TN_sum, FP_sum, FN_sum)
		F1 = evaluationF1score(TP_sum, TN_sum, FP_sum, FN_sum)
		RECALL = evaluationRecall(TP_sum, TN_sum, FP_sum, FN_sum)
		BA = evaluationBA(TP_sum, TN_sum, FP_sum, FN_sum)


		if AUC > best_auc:
			tailPredY = y_hat_all
			BEST_TAIL_AUC = AUC
			torch.save(model, "./result/longTail/Tail_IMC.pkl")

		kinaseAUC, kinaseAUPR, kinasePRECISION, kinaseRECALL, kinaseF1, kinaseBA = computeKinaseAUC.getEveryKinaseAUC(y_hat_all, kinaseAUC, kinaseAUPR, kinasePRECISION, kinaseRECALL, kinaseF1, kinaseBA)

		print('尾部 Epoch {}: TrainLoss {:.4f} TestLoss {:.4f} BA: {:.4f}  recall:{:.4f} precision:{:.4f} AUPR:{:.4f} F1:{:.4f} AUC:{:.4f} BESTAUC:{}'.format(
				epoch, loss, test_loss_all, BA, RECALL, PRECISION, F1, AUPR,AUC, BEST_TAIL_AUC))
		f.write('Tail,{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
				epoch, loss, test_loss_all, BA, RECALL, PRECISION, F1, AUPR,AUC, BEST_TAIL_AUC))

	f = open("./result/longTail/kinaseAUC.csv", "a")
	for i in range(len(list(kinaseAUC.values()))):
		f.write("{},{}\n".format(list(kinaseAUC.keys())[i], list(kinaseAUC.values())[i]))
	f.close

	f = open("./result/longTail/kinaseAUPR.csv", "a")
	for i in range(len(list(kinaseAUPR.values()))):
		f.write("{},{}\n".format(list(kinaseAUPR.keys())[i], list(kinaseAUPR.values())[i]))
	f.close

	f = open("./result/longTail/kinasePRECISION.csv", "a")
	for i in range(len(list(kinasePRECISION.values()))):
		f.write("{},{}\n".format(list(kinasePRECISION.keys())[i], list(kinasePRECISION.values())[i]))
	f.close

	f = open("./result/longTail/kinaseRECALL.csv", "a")
	for i in range(len(list(kinaseRECALL.values()))):
		f.write("{},{}\n".format(list(kinaseRECALL.keys())[i], list(kinaseRECALL.values())[i]))
	f.close

	f = open("./result/longTail/kinaseF1.csv", "a")
	for i in range(len(list(kinaseF1.values()))):
		f.write("{},{}\n".format(list(kinaseF1.keys())[i], list(kinaseF1.values())[i]))
	f.close

	f = open("./result/longTail/kinaseBA.csv", "a")
	for i in range(len(list(kinaseBA.values()))):
		f.write("{},{}\n".format(list(kinaseBA.keys())[i], list(kinaseBA.values())[i]))
	f.close
if __name__ == '__main__':
    main()
