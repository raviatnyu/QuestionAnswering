import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable

import layers.utils
#import utils

class Gate(nn.Module):

	def __init__(self, dict_args):
		super(Gate, self).__init__()
		self.sigmoidinputdim = dict_args['sigmoidinputdim']
		self.gateinputdim = dict_args['gateinputdim']
		self.linear = nn.Linear(self.sigmoidinputdim, self.gateinputdim)

	def forward(self, sigmoidinput, gateinput):
		#sigmoidinput: batch_size*sigmoidinputdim
		#gateinput: batch_size*gateinputdim
		sigmoidoutput = self.linear(sigmoidinput) #sigmoidoutput: batch_size*gateinputdim
		sigmoidoutput = functional.sigmoid(sigmoidoutput)
		return sigmoidoutput*gateinput #gateinput: batch_size*gateinputdim

if __name__=='__main__':
	gate = Gate({'sigmoidinputdim':20,'gateinputdim':10})
	print(gate(Variable(torch.randn(3,20)), Variable(torch.randn(3,10))))