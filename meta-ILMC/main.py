import torch
import os
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from model import IMC
from loadData import myDataset
from torch.utils.data import Dataset, DataLoader
from Metrics import *
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-3
BATCH_SIZE = 5000
DC = 167
DK = 343
DH = 16
EPOCHS = 20000
ALPHA = 1e-2
UPDATE_STEP = 3
UPDATE_STEP_TEST = 3
Q_length = 10

def train(IMC_model, optimizer, SLoader, QLoaderOne, QLoaderZero, init_para):
	#在s集上更新计算损失，更新模型
	vector_to_parameters(init_para, filter(lambda p: p.requires_grad, IMC_model.parameters()))
	lossFunction = nn.BCELoss().to(device)
	for i in range(UPDATE_STEP):
		for step, (compound, kinase, y) in enumerate(SLoader):
			predY = IMC_model(kinase.type(torch.float).to(device), compound.type(torch.float).to(device))
			loss_S = lossFunction(torch.squeeze(predY), torch.squeeze(y.to(device).type(torch.float)))
		#更新模型
		new_grad, new_params = update_params(loss_S, ALPHA, IMC_model)
		vector_to_parameters(new_params, filter(lambda p: p.requires_grad, IMC_model.parameters()))
	#计算Q集上的损失
	for step, data in enumerate(zip(QLoaderOne, QLoaderZero)):
		compound = torch.cat((data[0][0], data[1][0]), 0)
		kinase = torch.cat((data[0][1], data[1][1]), 0)
		y = torch.cat((data[0][2], data[1][2]), 0)
		break
	predY = IMC_model(kinase.type(torch.float).to(device), compound.type(torch.float).to(device))
	loss_Q = lossFunction(torch.squeeze(predY), torch.squeeze(y.to(device).type(torch.float)))
	vector_to_parameters(init_para, filter(lambda p: p.requires_grad, IMC_model.parameters()))
	return loss_Q
def test(IMC_model, optimizer, SLoader, QLoader, init_para):
	lossFunction = nn.BCELoss().to(device)
	for i in range(UPDATE_STEP_TEST):
		for step, (compound, kinase, y) in enumerate(SLoader):
			predY = IMC_model(kinase.type(torch.float).to(device), compound.type(torch.float).to(device))
			loss_S = lossFunction(torch.squeeze(predY), torch.squeeze(y.to(device).type(torch.float)))
		# 更新模型
		new_grad, new_params = update_params(loss_S, ALPHA, IMC_model)
		vector_to_parameters(new_params, filter(lambda p: p.requires_grad, IMC_model.parameters()))
	with torch.no_grad():
		for step, (compound, kinase, y) in enumerate(QLoader):
			predY = IMC_model(kinase.type(torch.float).to(device), compound.type(torch.float).to(device))
			y_scores = torch.squeeze(predY.to("cpu"))

			AUC = evaluationAuroc(y, y_scores)
			AUPR = evaluationAUPR(y, y_scores)

			y_scores[y_scores >= 0.5] = 1
			y_scores[y_scores < 0.5] = 0
			TP = int(sum(y[y_scores == 1]))
			FP = int(len(y[y_scores == 1]) - TP)
			FN = int(sum(y[y_scores == 0]))
			TN = int(len(y[y_scores == 0]) - FN)

			PRECISION = evaluationPrecision(TP, TN, FP, FN)
			F1 = evaluationF1score(TP, TN, FP, FN)
			RECALL = evaluationRecall(TP, TN, FP, FN)
			BA = evaluationBA(TP, TN, FP, FN)

			# print("AUC:{}".format(AUC))
	return AUC, AUPR, PRECISION, F1, RECALL, BA
def update_params(loss, update_lr, model):
	grads = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, model.parameters()))
	return parameters_to_vector(grads), parameters_to_vector(
		filter(lambda p: p.requires_grad, model.parameters())) - parameters_to_vector(grads) * update_lr
def main():
	bestLoss = 100
	resultPath = "./result/"
	trainTask = [1,282,378,40,331,19,255,131,258,342,348,324,149,167,172,212,302,386,108,174,280,106,233,273,300,277,142]
	# testTask = [140,48,215,275,161,261,242,67,69,235,247,206,216,56,269,317,71,334,319,265]
	testTask = set()
	f = pd.read_csv("./data/metaData/metaTailTrain.csv").values
	for i in range(len(f)):
		testTask.add(f[i][1])
	testTask = list(testTask)
	print(len(testTask))
	IMC_model = IMC(dimKinase = DK, dimCompound = DC, dimHidden = DH).to(device)
	# IMC_model = torch.load("./analysis_parameters_result/L_length/2/IMC.pkl")
	optimizer = optim.Adam(IMC_model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)  # 优化器
	#保存初始化参数
	init_para = parameters_to_vector(filter(lambda p: p.requires_grad, IMC_model.parameters()))
	BEST_AUC = {}
	BEST_AUPR = {}
	BEST_PRECISION = {}
	BEST_RECALL = {}
	BEST_F1 = {}
	BEST_BA = {}
	trainLoadersS = {}
	trainLoadersQ_one = {}
	trainLoadersQ_zero = {}
	print("============================读取文件============================")
	print("开始读取元训练任务")
	for task in trainTask:
		print("开始读取任务：{}".format(task))
		trainLoadersS[task] = DataLoader(myDataset(train="longTrain", task=task),
		                                 batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
		trainLoadersQ_one[task] = DataLoader(myDataset(train="longTestOne", task=task),
		                                     batch_size=Q_length, shuffle=True, drop_last=False, num_workers=0)
		trainLoadersQ_zero[task] = DataLoader(myDataset(train="longTestZero", task=task),
		                                     batch_size=Q_length, shuffle=True, drop_last=False, num_workers=0)
		print("任务：{} 读取完毕".format(task))
	print("元训练任务读取完毕")
	print("开始读取元测试任务")
	testLoadersS = {}
	testLoadersQ = {}
	for task in testTask:
		print("开始读取任务：{}".format(task))
		testLoadersS[task] = DataLoader(myDataset(train="tailTrain", task=task),
		                                batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
		testLoadersQ[task] = DataLoader(myDataset(train="tailTest", task=task),
		                                batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
		print("任务：{} 读取完毕".format(task))
	print("元测试任务读取完毕")
	print("============================读取完毕============================\n\n\n")
	print("============================开始训练============================")
	for i in range(len(testTask)):
		BEST_AUC[testTask[i]] = 0
		BEST_AUPR[testTask[i]] = 0
		BEST_PRECISION[testTask[i]] = 0
		BEST_RECALL[testTask[i]] = 0
		BEST_F1[testTask[i]] = 0
		BEST_BA[testTask[i]] = 0

	for epoch in range(EPOCHS):
		losses_Q = torch.tensor([0.0]).to(device)
		for task in trainTask:
			loss_Q = torch.tensor([0.0]).to(device)
			SLoader = trainLoadersS[task]
			QLoaderOne = trainLoadersQ_one[task]
			QLoaderZero = trainLoadersQ_zero[task]
			loss_Q += train(IMC_model, optimizer, SLoader, QLoaderOne, QLoaderZero, init_para)
			if task == trainTask[0]:
				losses_Q = loss_Q
			else:
				losses_Q = torch.cat((losses_Q, loss_Q), 0)
		losses_Q = torch.sum(losses_Q)
		# losses_Q = losses_Q / len(trainTask)
		optimizer.zero_grad()
		losses_Q.backward()
		optimizer.step()
		init_para = torch.clone(parameters_to_vector(filter(lambda p: p.requires_grad, IMC_model.parameters())))
		print("EPOCH:{}   LOSS_Q:{:.4f}".format(epoch, losses_Q.item()))
		if losses_Q.item() < bestLoss:
			torch.save(IMC_model, resultPath + "IMC.pkl")
			bestLoss = losses_Q.item()
		fileAUC = open(resultPath + "AUCresult.csv", "a")
		fileAUC.write("{}".format(epoch))

		fileAUPR = open(resultPath + "AUPRresult.csv", "a")
		fileAUPR.write("{}".format(epoch))

		filePRECISION = open(resultPath + "PRECISIONresult.csv", "a")
		filePRECISION.write("{}".format(epoch))

		fileRECALL = open(resultPath + "RECALLresult.csv", "a")
		fileRECALL.write("{}".format(epoch))

		fileF1 = open(resultPath + "F1result.csv", "a")
		fileF1.write("{}".format(epoch))

		fileBA = open(resultPath + "BAresult.csv", "a")
		fileBA.write("{}".format(epoch))

		for task in testTask:
			SLoader = testLoadersS[task]
			QLoader = testLoadersQ[task]
			AUC, AUPR, PRECISION, F1, RECALL, BA = test(IMC_model, optimizer, SLoader, QLoader, init_para)
			# print("---------------------------------------")
			if AUC > BEST_AUC[task]:
				BEST_AUC[task] = AUC
				if os.path.exists(resultPath + "model/"+str(task)):
					pass
				else:
					os.mkdir(resultPath + "model/"+str(task))
				torch.save(IMC_model, resultPath + "model/"+str(task)+"/IMC.pkl")
			if AUPR > BEST_AUPR[task]:
				BEST_AUPR[task] = AUPR
			if PRECISION > BEST_PRECISION[task]:
				BEST_PRECISION[task] = PRECISION
			if RECALL > BEST_RECALL[task]:
				BEST_RECALL[task] = RECALL
			if F1 > BEST_F1[task]:
				BEST_F1[task] = F1
			if BA > BEST_BA[task]:
				BEST_BA[task] = BA
			fileAUC.write(",{},{}".format(task, AUC))
			fileAUPR.write(",{},{}".format(task, AUPR))
			filePRECISION.write(",{},{}".format(task, PRECISION))
			fileRECALL.write(",{},{}".format(task, RECALL))
			fileF1.write(",{},{}".format(task, F1))
			fileBA.write(",{},{}".format(task, BA))
		fileAUC.write("\n")
		fileAUC.close()
		fileAUPR.write("\n")
		fileAUPR.close()
		filePRECISION.write("\n")
		filePRECISION.close()
		fileRECALL.write("\n")
		fileRECALL.close()
		fileF1.write("\n")
		fileF1.close()
		fileBA.write("\n")
		fileBA.close()

		f=open(resultPath + "Loss_Q.csv", "a")
		f.write("{},{}\n".format(epoch, losses_Q.item()))
		f.close()
		if (epoch + 1) % 1000 == 0:
			f = open(resultPath + "auc/AUCbestResult"+str(epoch)+".csv", "a")
			for i in range(len(list(BEST_AUC.values()))):
				f.write("{},{}\n".format(list(BEST_AUC.keys())[i], list(BEST_AUC.values())[i]))
			f.close()

			f = open(resultPath + "aupr/AUPRbestResult" + str(epoch) + ".csv", "a")
			for i in range(len(list(BEST_AUPR.values()))):
				f.write("{},{}\n".format(list(BEST_AUPR.keys())[i], list(BEST_AUPR.values())[i]))
			f.close()

			f = open(resultPath + "precision/PRECISIONbestResult" + str(epoch) + ".csv", "a")
			for i in range(len(list(BEST_PRECISION.values()))):
				f.write("{},{}\n".format(list(BEST_PRECISION.keys())[i], list(BEST_PRECISION.values())[i]))
			f.close()

			f = open(resultPath + "recall/RECALLbestResult" + str(epoch) + ".csv", "a")
			for i in range(len(list(BEST_RECALL.values()))):
				f.write("{},{}\n".format(list(BEST_RECALL.keys())[i], list(BEST_RECALL.values())[i]))
			f.close()

			f = open(resultPath + "f1/F1bestResult" + str(epoch) + ".csv", "a")
			for i in range(len(list(BEST_F1.values()))):
				f.write("{},{}\n".format(list(BEST_F1.keys())[i], list(BEST_F1.values())[i]))
			f.close()

			f = open(resultPath + "ba/BAbestResult" + str(epoch) + ".csv", "a")
			for i in range(len(list(BEST_BA.values()))):
				f.write("{},{}\n".format(list(BEST_BA.keys())[i], list(BEST_BA.values())[i]))
			f.close()

if __name__ == '__main__':
	main()