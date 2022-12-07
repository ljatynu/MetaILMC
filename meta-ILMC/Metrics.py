from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
def evaluationAuroc(y_true, y_hat_all):
	try:
		AUC = roc_auc_score(y_true, y_hat_all)
	except:
		AUC = 0
	return AUC
def evaluationRecall(TP, TN, FP, FN):
	recall = TP / (TP + FN) if TP != 0 else 0
	return recall
def evaluationPrecision(TP, TN, FP, FN):
	precision = TP / (TP + FP) if TP != 0 else 0
	return precision
def evaluationF1score(TP, TN, FP, FN):
	precision = evaluationPrecision(TP, TN, FP, FN)
	recall = evaluationRecall(TP, TN, FP, FN)
	F1SCORE = (2 * precision * recall) / (precision + recall) if TP != 0 else 0
	return F1SCORE
def evaluationBA(TP, TN, FP, FN):
	if (TP + FN) == 0 or (FP + TN) == 0:
		BA = 0
	else:
		BA = 0.5 * (TP / (TP + FN) + TN / (FP + TN))
	return BA
def evaluationAUPR(y_true, y_hat_all):
	precision, recall, thresholds = precision_recall_curve(y_true, y_hat_all)
	AUPR = auc(recall, precision)
	return AUPR