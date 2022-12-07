import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,roc_curve
import random
from Metrics import *
path = "./data/longTail/"

def getKinaseID():
	train = pd.read_csv(path + "longTailTrain.csv").values
	longTest = pd.read_csv(path + "longTest.csv").values
	tailTest = pd.read_csv(path + "tailTest.csv").values


	tailKinaseID = {}
	for i in range(len(tailTest)):
		tailKinaseID[tailTest[i][1]] = tailKinaseID.get(tailTest[i][1], 0) + 1
	tailKinaseID = list(set(list(tailKinaseID.keys())))
	trainKinaseFre = {}
	for i in range(len(train)):
		trainKinaseFre[train[i][0]] = trainKinaseFre.get(train[i][0], 0) + 1

	kinaseAUC,kinaseAUPR,kinasePRECISION,kinaseRECALL,kinaseF1,kinaseBA = {},{},{},{},{},{}
	for kinase in tailKinaseID:
		kinaseAUC[kinase] = 0
		kinaseAUPR[kinase] = 0
		kinasePRECISION[kinase] = 0
		kinaseRECALL[kinase] = 0
		kinaseF1[kinase] = 0
		kinaseBA[kinase] = 0
	return kinaseAUC, kinaseAUPR, kinasePRECISION, kinaseRECALL, kinaseF1, kinaseBA


def getEveryKinaseAUC(tailPredY, kinaseAUC, kinaseAUPR, kinasePRECISION, kinaseRECALL, kinaseF1, kinaseBA):
	train = pd.read_csv(path + "longTailTrain.csv").values
	longTest = pd.read_csv(path + "longTest.csv").values
	tailTest = pd.read_csv(path + "tailTest.csv").values

	tailTest = np.concatenate([tailTest, np.array(tailPredY).reshape(-1, 1)], 1)  # 将预测数据拼接到tailTest中
	tailKinaseID = {}
	for i in range(len(tailTest)):
		tailKinaseID[tailTest[i][1]] = tailKinaseID.get(tailTest[i][1], 0) + 1
	tailKinaseID = list(set(list(tailKinaseID.keys())))
	trainKinaseFre = {}
	for i in range(len(train)):
		trainKinaseFre[train[i][0]] = trainKinaseFre.get(train[i][0], 0) + 1

	for kinase in tailKinaseID:
		y = tailTest[tailTest[:, 1] == kinase][:, 2]
		predY = tailTest[tailTest[:, 1] == kinase][:, 3]

		predY[predY >= 0.5] = 1
		predY[predY < 0.5] = 0
		TP = sum(y[predY == 1])
		FP = len(y[predY == 1]) - TP
		FN = sum(y[predY == 0])
		TN = len(y[predY == 0]) - FN

		AUC = evaluationAuroc(y, predY)  # 测试集中只有一个类别时，没有auc
		AUPR = evaluationAUPR(y, predY)
		PRECISION = evaluationPrecision(TP, TN, FP, FN)
		F1 = evaluationF1score(TP, TN, FP, FN)
		RECALL = evaluationRecall(TP, TN, FP, FN)
		BA = evaluationBA(TP, TN, FP, FN)

		if AUC > kinaseAUC[kinase]:
			kinaseAUC[kinase] = AUC
		if AUPR > kinaseAUPR[kinase]:
			kinaseAUPR[kinase] = AUPR
		if PRECISION > kinasePRECISION[kinase]:
			kinasePRECISION[kinase] = PRECISION
		if RECALL > kinaseRECALL[kinase]:
			kinaseRECALL[kinase] = RECALL
		if F1 > kinaseF1[kinase]:
			kinaseF1[kinase] = F1
		if BA > kinaseBA[kinase]:
			kinaseBA[kinase] = BA
	return kinaseAUC, kinaseAUPR, kinasePRECISION, kinaseRECALL, kinaseF1, kinaseBA
if __name__ == '__main__':
	kinaseAUC = getKinaseID()
	# 使用readline()读文件
	f = open(path + "result.txt")
	while True:
		line = f.readline()
		if line:
			if line[:9] == "tailPredY":
				tailPredY = eval(line.split("=")[1])
		else:
			break
	f.close()
	kinaseAUC = getEveryKinaseAUC(tailPredY, kinaseAUC)
	print(kinaseAUC)