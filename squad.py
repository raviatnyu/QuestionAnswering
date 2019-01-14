from __future__ import print_function
# This Python file uses the following encoding: utf-8
import os
import sys
import json
import pickle
import nltk
#nltk.download('punkt')
#sys.path.append('~/nltk_data/')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
import torch.optim as optim

from bidaf import BiDAF
from rnet import RNet
import input.dataloader as loader
import xeval as evaluator
from data.preprocessutils import *
from data.preprocess import *
from convert import convert

USE_CUDA = False
#reload(sys)
#sys.setdefaultencoding('utf-8')

def evaluate(dataloader, model):
	model = model.eval()
	predictiondict = {}
	predictiondictadv = {}
	for batch in dataloader:
		if True:
			'''passage, passage_lengths, passagechars, passagechar_lengths,\
			question, question_lengths, questionchars, questionchar_lengths,\
			instance_ids, passage_tokens = convert(batch, USE_CUDA, 'eval')'''
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

			best_span = model.find_span(start_index_log_probabilities.data.cpu(), end_index_log_probabilities.data.cpu())	

			for index, span in enumerate(best_span):
				instance_id = instance_ids[index]
				current_passage_tokens = passage_tokens[index]
				current_passage_length = len(current_passage_tokens)
				answer = ''
				for i in range(int(span[0]), int(span[1])+1):
					if i < current_passage_length:
						if current_passage_tokens[i] in extra_token_splits:
							answer = answer.strip() + current_passage_tokens[i]
						else:
							answer = answer + current_passage_tokens[i] + ' '
					else:
						printf("Masking Error for {}",instance_id)
				answer = answer.strip()
				predictiondict[instance_id] = answer
				predictiondictadv[instance_id] = post_process(answer)
				#print(post_process(answer))

	predictionadvjson = json.loads(json.dumps(predictiondictadv))
	return predictionadvjson

def test(modelpath, modelname, devfile, predfile):
	def get_model(modeldict, modelname):
		if USE_CUDA:
			checkpoint = torch.load(modeldict)
		else:
			checkpoint = torch.load(modeldict, map_location=lambda storage, loc: storage)
		dict_args = checkpoint['dict_args']
		if modelname == 'bidaf':
			model = BiDAF(dict_args)
		elif modelname == 'rnet':
			model = RNet(dict_args)
		if USE_CUDA:
			model = model.cuda()
		model.load_state_dict(checkpoint['state_dict'])
		return model
	def get_vocab(modelpath):
		vocabfile = open(os.path.join(modelpath, 'vocab.pkl'), 'rb')
		vocab = pickle.load(vocabfile)
		glovefile = open(os.path.join(modelpath, 'glove.pkl'), 'rb')
		glove = pickle.load(glovefile)
		return vocab, glove

	if modelname == 'bidaf':
		modeldict = open(os.path.join(modelpath, 'bidaf.pth'), 'rb')
		model = get_model(modeldict, 'bidaf')
	elif modelname == 'rnet':
		modeldict = open(os.path.join(modelpath, 'rnet.pth'), 'rb')
		model = get_model(modeldict, 'rnet')

	vocab, glove = get_vocab(modelpath)

	eval_batch_size = 10
	devjsons = [devfile]
	outputpath = predfile

	files = [os.path.join(os.getcwd(),file) for file in devjsons]
	dataloader = loader.get_test_data(files, vocab, glove, eval_batch_size)
	predjson = evaluate(dataloader, model)
	file = open(os.path.join(os.getcwd(), predfile), 'w')
	json.dump(predjson, file)
	file.close()
	return

if __name__=='__main__':
	devfile = sys.argv[1]
	predfile = sys.argv[2]

	devdata = json.dumps(process_data(devfile))
	testfilename = 'dev.json'
	devfile = open(testfilename,'w')
	devfile.write(devdata)
	devfile.close()
	
	#test(os.path.join(os.getcwd(),'src/models/bidirLSTM10060drop0p1/'), 'rnet', testfilename, predfile)
	test(os.path.join(os.getcwd(),'src/models/bidafLSTM100100drop0p2/'), 'bidaf', testfilename, predfile)




