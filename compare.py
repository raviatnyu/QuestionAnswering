from __future__ import print_function
# This Python file uses the following encoding: utf-8
import os
import sys
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
import torch.optim as optim

from bidaf import BiDAF
from rnet import RNet
import input.dataloader as loader
import evaluate as evaluator
from data.preprocessutils import *

USE_CUDA = False
#reload(sys)
#sys.setdefaultencoding('utf-8')

def compare(bidafmodelpath, rnetmodelpath):
	def get_model(modeldict, modelname):
		checkpoint = torch.load(modeldict, map_location=lambda storage, loc: storage)
		dict_args = checkpoint['dict_args']
		if modelname == 'bidaf':
			model = BiDAF(dict_args)
		elif modelname == 'rnet':
			model = RNet(dict_args)
		model.load_state_dict(checkpoint['state_dict'])
		return model
	def get_vocab(modelpath):
		vocabfile = open(os.path.join(modelpath, 'vocab.pkl'), 'rb')
		vocab = pickle.load(vocabfile)
		glovefile = open(os.path.join(modelpath, 'glove.pkl'), 'rb')
		glove = pickle.load(glovefile)
		return vocab, glove

	bidafmodeldict = open(os.path.join(bidafmodelpath, 'bidaf.pth'), 'rb')
	rnetmodeldict = open(os.path.join(rnetmodelpath, 'rnet.pth'), 'rb')

	bidafmodel = get_model(bidafmodeldict, 'bidaf')
	rnetmodel = get_model(rnetmodeldict, 'rnet')
	bidafvocab, bidafglove = get_vocab(bidafmodelpath)
	#rnetvocab, rnetglove = get_vocab(rnetmodelpath)

	vocab = bidafvocab
	glove = bidafglove
	eval_batch_size = 1
	file_names = ['input/dev50.json']
	files = [os.path.join(os.getcwd(),file) for file in file_names]
	dataloader = loader.get_test_data(files, vocab, glove, eval_batch_size)
	groundtruth = json.load(open(os.path.join(os.getcwd(),'data/dev-v1.1.json')))

	evaluate(dataloader, bidafmodel, groundtruth)
	evaluate(dataloader, rnetmodel, groundtruth)
	return

def evaluate(dataloader, model, groudtruthjson):
	model = model.eval()
	predictiondictadv = {}
	count = 0
	for batch in dataloader:
		if True:
			count = count + 1
			passage = Variable(torch.FloatTensor(batch[0]), volatile = True)
			passage_lengths = torch.LongTensor(batch[1])
			passagechars = Variable(torch.LongTensor(batch[2]), volatile = True)
			passagechar_lengths = torch.LongTensor(batch[3])
			question = Variable(torch.FloatTensor(batch[4]), volatile = True)
			question_lengths = torch.LongTensor(batch[5])
			questionchars = Variable(torch.LongTensor(batch[6]), volatile = True)
			questionchar_lengths = torch.LongTensor(batch[7])
			instance_ids = batch[9]
			passage_tokens = batch[10]	
			question_tokens = batch[11]	
			answer_span = torch.LongTensor(batch[8])

			#if(question_tokens[0][0] == 'What'):
			if True:
				if USE_CUDA:
					passage = passage.cuda()
					passage_lengths = passage_lengths.cuda()
					passagechars = passagechars.cuda()
					passagechar_lengths = passagechar_lengths.cuda()
					question = question.cuda()
					question_lengths = question_lengths.cuda()
					questionchars = questionchars.cuda()
					questionchar_lengths = questionchar_lengths.cuda()	
				start_index_log_probabilities, end_index_log_probabilities, attention_weights = \
					model(passage, passage_lengths, passagechars, passagechar_lengths,\
						question, question_lengths, questionchars, questionchar_lengths)	
				if instance_ids[0] == '56d723ad0d65d214001983b4':
				#if True:
					best_span = model.find_span(start_index_log_probabilities.data.cpu(), end_index_log_probabilities.data.cpu())
					print(passage_tokens)
					print(question_tokens)					
					print(best_span)
					print(answer_span)
					print(instance_ids)
					fig = plt.figure()
					ax = fig.add_subplot(111)
					cax = ax.matshow(attention_weights.squeeze().data.numpy(), cmap='bone', aspect='auto')
					fig.colorbar(cax)
					plt.show()
					ax.set_xticklabels(passage_tokens)
					ax.set_yticklabels(question_tokens)
					ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
					ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
					return

if __name__=='__main__':
	compare('models/bidafLSTM100100drop0p2/', 'models/bidirLSTM10060drop0p1')

#56d723ad0d65d214001983b4
#56e12005cd28a01900c67617 - 'what'
#56e77cee00c9c71400d771aa
#56f884cba6d7ea1400e17707

#56de49a8cffd8e1900b4b7a8 failed case
#56de49a8cffd8e1900b4b7a9 failed case

#25
#56e74af500c9c71400d76f68 first one in 'who'

#How should i include
#Where

#Abiguous
#56e125b6e3433e1400422c6c