import torch
import torch.nn as nn
import torch.nn.functional as F
class IMC(nn.Module):
	def __init__(self, dimKinase = 343, dimCompound = 166, dimHidden = 64):
		super().__init__()
		self.DK = dimKinase
		self.DC = dimCompound
		self.DH = dimHidden

		self.WK1 = nn.Linear(self.DK, 128)
		self.WK2 = nn.Linear(128, 64)
		self.WK3 = nn.Linear(64, self.DH)


		self.WC1 = nn.Linear(self.DC, 128)
		self.WC2 = nn.Linear(128, 64)
		self.WC3 = nn.Linear(64, self.DH)
	# def reset_parameters(self):
	# 	for m in self.modules():
	# 		if isinstance(m, nn.Linear):
	# 			nn.init.xavier_normal_(m.weight.data, gain=1)
	def reset_parameters(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
	def forward(self, kinase, compound):
		pre = torch.zeros(kinase.shape[0])

		Hkinase = torch.tanh(self.WK1(kinase))#激酶隐藏层表示
		Hkinase1 = torch.tanh(self.WK2(Hkinase))
		Hkinase2 = torch.tanh(self.WK3(Hkinase1))

		Hcompound = torch.tanh(self.WC1(compound))#化合物隐藏层表示
		Hcompound1 = torch.tanh(self.WC2(Hcompound))
		Hcompound2 = torch.tanh(self.WC3(Hcompound1))

		output = torch.sum(Hkinase2 * Hcompound2, 1)
		# output = Hcompound2.mm(Hkinase2.T)
		return torch.unsqueeze(torch.sigmoid(output), 1)
if __name__ == '__main__':#测试
	model = IMC(dimKinase = 343, dimCompound = 166, dimHidden = 2)
	# model.reset_parameters()
	a = torch.randn(10, 343)
	b = torch.randn(10, 166)
	y = model(a, b)
	print(y)