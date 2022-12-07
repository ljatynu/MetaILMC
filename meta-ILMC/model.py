import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
class IMC(nn.Module):
	def __init__(self, dimKinase = 343, dimCompound = 166, dimHidden = 64):
		super().__init__()
		self.DK = dimKinase
		self.DC = dimCompound
		self.DH = dimHidden

		self.WK1 = nn.Linear(self.DK, 128)
		self.WK2 = nn.Linear(128, 64)
		self.WK3 = nn.Linear(64, 32)
		self.WK4 = nn.Linear(32, self.DH)


		self.WC1 = nn.Linear(self.DC, 128)
		self.WC2 = nn.Linear(128, 64)
		self.WC3 = nn.Linear(64, 32)
		self.WC4 = nn.Linear(32, self.DH)
	def reset_parameters(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
	def forward(self, kinase, compound):

		Hkinase = torch.tanh(self.WK1(kinase))#激酶隐藏层表示
		Hkinase1 = torch.tanh(self.WK2(Hkinase))
		Hkinase2 = torch.tanh(self.WK3(Hkinase1))
		Hkinase3 = torch.tanh(self.WK4(Hkinase2))

		Hcompound = torch.tanh(self.WC1(compound))#化合物隐藏层表示
		Hcompound1 = torch.tanh(self.WC2(Hcompound))
		Hcompound2 = torch.tanh(self.WC3(Hcompound1))
		Hcompound3 = torch.tanh(self.WC4(Hcompound2))

		output = torch.sum(Hkinase3 * Hcompound3, 1)
		return torch.unsqueeze(torch.sigmoid(output), 1)
if __name__ == '__main__':#测试
	model = IMC(dimKinase = 343, dimCompound = 166, dimHidden = 2)
	a = torch.randn(10, 343)
	b = torch.randn(10, 166)
	y = model(a, b)
	print(y)