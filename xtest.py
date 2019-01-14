from __future__ import print_function
# This Python file uses the following encoding: utf-8
import os
import sys
import json
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
import torch.optim as optim

from nets import BiDAF
from nets import RNet
from input import loader
import xeval as evaluator
from data import post_process

USE_CUDA = False
#reload(sys)
#sys.setdefaultencoding('utf-8')
import nltk
nltk.download('punkt')

def evaluate(dataloader, model, groudtruthjson):
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

			start_index_log_probabilities, end_index_log_probabilities''', attention_weights''' = \
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

	predictionjson = json.loads(json.dumps(predictiondict))
	resultjson = evaluator.evaluate(groudtruthjson['data'], predictionjson)
	predictionadvjson = json.loads(json.dumps(predictiondictadv))
	resultadvjson = evaluator.evaluate(groudtruthjson['data'], predictionadvjson)
	return resultadvjson['exact_match'], resultadvjson['f1'],\
			resultadvjson['emids'], resultadvjson['nonemids'], predictionadvjson

def test(modelpath, modelname):
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

	eval_batch_size = 30
	
	devjsons = ['dev_how.json','dev_len1.json','dev_len10.json','dev_len100.json','dev_len2.json','dev_len5.json','dev_misc.json','dev_what.json','dev_when.json','dev_where.json','dev_which.json','dev_who.json','dev_why.json']
	groudtruthjsons = ['dev-v1.1_how.json','dev-v1.1_len1.json','dev-v1.1_len10.json','dev-v1.1_len100.json','dev-v1.1_len2.json','dev-v1.1_len5.json','dev-v1.1_misc.json','dev-v1.1_what.json','dev-v1.1_when.json','dev-v1.1_where.json','dev-v1.1_which.json','dev-v1.1_who.json','dev-v1.1_why.json']

	outputpath = 'output/' + modelpath.split('/')[1]

	for devjson, groudtruthjson in zip(devjsons, groudtruthjsons):
		file_names = ['input/' + devjson]
		groudtruthjson = 'data/' + groudtruthjson
		files = [os.path.join(os.getcwd(),file) for file in file_names]
		dataloader = loader.get_test_data(files, vocab, glove, eval_batch_size)
		groundtruth = json.load(open(os.path.join(os.getcwd(),'data/dev-v1.1.json')))
		em, f1, emids, nonemids, predjson = evaluate(dataloader, model, groundtruth)
		print('file = {}, em = {}, f1 = {}'.format(devjson, em, f1))
		filename = 'pred' + devjson
		file = open(os.path.join(outputpath, filename), 'w')
		json.dump(predjson, file)
		file.close()
	return

if __name__=='__main__':
	test('models/bidafLSTM100100drop0p2/', 'bidaf')




