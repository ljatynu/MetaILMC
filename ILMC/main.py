from torch.utils.data import DataLoader, random_split
from getData import myDataset
from model import IMC
from torch import nn, optim
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import torch.functional as F
import mergedData
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
	lossFunction = nn.BCELoss().to(device) #自定义的损失函数
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
	BESTAUC = 0
	BESTBA = 0
	BESTF1 = 0
	BESTRECALL = 0
	BESTPRECISION = 0
	BESTAUPR = 0

	fold = 9

	trainSet = mergedData.myDataset(train = True, fold = fold)
	testSet = mergedData.myDataset(train = False, fold = fold)
	print(fold)
	print("======================数据集大小=========================")
	print("训练集大小为：{}".format(len(trainSet)))
	print("测试集大小为：{}".format(len(testSet)))
	print("======================开始训练！=========================")
	trainData = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
	testData = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
	model = IMC(dimKinase = DK, dimCompound = DC, dimHidden = DH).to(device) #获取模型
	# model.reset_parameters()

	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0) #优化器
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
	trainLossList, testLoss, testACC, testAUC = [], [], [], []
	f = open("./result/tenFold/" + str(fold) + "/EPOCH_result.csv", "a")
	f.write("epoch,loss,test_loss_all,balanceACC,AUPR,recall,precision,F1SCORE,AUC,BESTAUC,BESTBA,BESTAUPR,BESTF1,BESTRECALL,BESTPRECISION\n")
	f.close()
	for epoch in range(EPOCHS):
		loss = 0
		for i, (compound, kinase, y) in enumerate(trainData):
			loss += train(compound.to(device), kinase.to(device), torch.unsqueeze(y.to(device), 1), model, optimizer)
		trainLossList.append(loss)
		scheduler.step()
		TP_sum, TN_sum, FN_sum, FP_sum = 0., 0., 0., 0.
		test_loss_all = 0
		y_true, y_hat_all = [], []
		for i, (compound, kinase, y) in enumerate(testData):
			TP, TN, FN, FP, y_hat, test_loss = test(compound.to(device), kinase.to(device), y.to(device), model)
			test_loss_all += test_loss
			TP_sum += TP
			TN_sum += TN
			FN_sum += FN
			FP_sum += FP
			y_true = y_true + list(np.array(y))
			y_hat_all = y_hat_all + list(y_hat)

		print("TP:{}  TN:{}  FP:{}  FN:{}".format(TP_sum, TN_sum, FP_sum, FN_sum))
		AUC = evaluationAuroc(y_true, y_hat_all)
		balanceACC = evaluationBA(TP_sum, TN_sum, FP_sum, FN_sum)
		recall = evaluationRecall(TP_sum, TN_sum, FP_sum, FN_sum)
		precision = evaluationPrecision(TP_sum, TN_sum, FP_sum, FN_sum)
		AUPR = evaluationAUPR(y_true, y_hat_all)
		F1SCORE = evaluationF1score(TP_sum, TN_sum, FP_sum, FN_sum)

		if AUC > BESTAUC:
			BESTAUC = AUC
			torch.save(model, "./result/tenFold/" + str(fold) + "/IMC.pkl")
		if balanceACC > BESTBA:
			BESTBA = balanceACC
		if recall > BESTRECALL:
			BESTRECALL = recall
		if precision > BESTPRECISION:
			BESTPRECISION = precision
		if AUPR > BESTAUPR:
			BESTAUPR = AUPR
		if F1SCORE > BESTF1:
			BESTF1 = F1SCORE

		print('Epoch {}: TrainLoss {:.4f} TestLoss {:.4f} BA: {:.4f} AUPR: {:.4f} recall:{:.4f} precision:{:.4f} F1:{:.4f} AUC:{:.4f} BESTAUC:{:.4f}'.format(
				epoch, loss, test_loss_all, balanceACC, AUPR, recall, precision, F1SCORE, AUC, BESTAUC))
		f = open("./result/tenFold/" + str(fold) + "/EPOCH_result.csv", "a")
		f.write("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
			epoch, loss, test_loss_all, balanceACC, AUPR, recall, precision, F1SCORE, AUC, BESTAUC,BESTBA,BESTAUPR,BESTF1,BESTRECALL,BESTPRECISION))
		f.close()
if __name__ == '__main__':
    main()