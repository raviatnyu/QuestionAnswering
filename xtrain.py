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

from bidaf import BiDAF
from rnet import RNet
from xhelper import *
import input.dataloader as loader
import xeval as evaluator
from data.preprocessutils import *

USE_CUDA = True
reload(sys)
sys.setdefaultencoding('utf-8')

def evaluate(dataloader, model, groudtruthjson):
	model = model.eval()
	predictiondict = {}
	predictiondictadv = {}
	for superbatch in dataloader:
		superbatch_size = len(superbatch[0])
		subbatchzise = get_eval_subbatch(superbatch[0][0])
		for j in range(int(superbatch_size/subbatchzise)):
			start_size = j*subbatchzise
			end_size = (j+1)*subbatchzise
			if end_size > superbatch_size:
				end_size = superbatch_size
			batch = []
			for k in range(len(superbatch)):
				batch.append(superbatch[k][start_size : end_size])

			passage, passage_lengths, passagechars, passagechar_lengths,\
			question, question_lengths, questionchars, questionchar_lengths,\
			instance_ids, passage_tokens = convert(batch, USE_CUDA, 'eval')

			start_index_log_probabilities, end_index_log_probabilities = \
				model(passage, passage_lengths, passagechars, passagechar_lengths,\
					question, question_lengths, questionchars, questionchar_lengths)	

			best_span = model.find_span(start_index_log_probabilities.data.cpu(), end_index_log_probabilities.data.cpu())	

			for index, span in enumerate(best_span):
				instance_id = instance_ids[index]
				current_passage_tokens = passage_tokens[index]
				answer = getanswer(instance_id, current_passage_tokens, span)
				predictiondict[instance_id] = answer
				predictiondictadv[instance_id] = post_process(answer)

	predictionjson = json.loads(json.dumps(predictiondict))
	resultjson = evaluator.evaluate(groudtruthjson['data'], predictionjson)
	predictionadvjson = json.loads(json.dumps(predictiondictadv))
	resultadvjson = evaluator.evaluate(groudtruthjson['data'], predictionadvjson)
	print('Basic Scores = {}, {}'.format(resultjson['exact_match'], resultjson['f1']))
	print('Advanced Scores = {}, {}'.format(resultadvjson['exact_match'], resultadvjson['f1']))
	return resultadvjson['exact_match'], resultadvjson['f1']


def train_rnet():
	cur_dir = os.getcwd()
	glove_dir = '../glove'
	glove_filename = 'glove.6B.100d.txt'
	glove_filepath = os.path.join(glove_dir, glove_filename)

	train_batch_size = 60
	eval_batch_size = 60
	batch_size = train_batch_size

	file_names = ['input/train.json']
	files = [os.path.join(cur_dir,file) for file in file_names]
	train_dataloader, vocab, glove = loader.get_train_data(files, glove_filepath, train_batch_size)

	file_names = ['input/dev.json']
	files = [os.path.join(cur_dir,file) for file in file_names]
	dev_dataloader = loader.get_test_data(files, vocab, glove, eval_batch_size)

	traingroundtruth = json.load(open(os.path.join(cur_dir,'data/train-v1.1.json')))
	devgroundtruth = json.load(open(os.path.join(cur_dir,'data/dev-v1.1.json')))

	save_dir = 'models/bidirLSTM10040drop0p0/'

	glovefile = open(os.path.join(save_dir, 'glove.pkl'), 'wb')
	pickle.dump(glove, glovefile)
	glovefile.close()

	vocabfile = open(os.path.join(save_dir, 'vocab.pkl'), 'wb')
	pickle.dump(vocab, vocabfile)
	vocabfile.close()

	dict_args = {
				 'use_charemb': True,
				 'charvocab_size':len(vocab.char2index),
				 'charemb_dim':50,
				 'charemb_rnn_hdim':25,
				 'charemb_rnn_type':'LSTM',
				 'charemb_padix':0,
				 'wordemb_dim':100,
				 'contextemb_rnn_hdim':40,
				 'contextemb_rnn_type':'LSTM',
				 'contextemb_num_layers':3,
				 'gated_attention_similarity_function': 'ProjectionSimilaritySharedWeights',
				 'use_bidirectional': True,
				 'use_gating': True,
				 'use_selfmatching': True,
				 'gated_attentio_rnn_type': 'LSTM',
				 'self_matching_similarity_function': 'ProjectionSimilaritySharedWeights',
				 'modelinglayer_rnn_type':'LSTM',
				 'modelinglayer_num_layers':1,
				 'question_attention_similarity_function': 'ProjectionSimilaritySharedWeights',
				 'pointer_network_similarity_function': 'ProjectionSimilaritySharedWeights',
				 'pointer_network_rnn_type':'GRU',
				 'dropout_rate':0
				}
	rnet = RNet(dict_args)

	num_epochs = 20
	learning_rate = 1.0
	criterion = nn.NLLLoss()  
	optimizer = optim.Adadelta(rnet.parameters(), lr=learning_rate, rho=0.95, eps=1e-06, weight_decay=0)
	if USE_CUDA:
		rnet = rnet.cuda()
		criterion = criterion.cuda()
	best_val_f1 = 0.0

	for epoch in range(num_epochs):
		predictiondict = {}
		predictiondictadv = {}
		for i,superbatch in enumerate(train_dataloader):
			superbatch_size = len(superbatch[0])
			subbatchzise = get_rnet_subbatch(superbatch[0][0])
			#print(subbatchzise)
			for j in range(int(superbatch_size/subbatchzise)):
				start_size = j*subbatchzise
				end_size = (j+1)*subbatchzise
				if end_size > superbatch_size:
					end_size = superbatch_size
				batch = []
				for k in range(len(superbatch)):
					batch.append(superbatch[k][start_size : end_size])

				passage, passage_lengths, passagechars, passagechar_lengths,\
				question, question_lengths, questionchars, questionchar_lengths,\
				start_indices, end_indices, instance_ids, passage_tokens\
				= convert(batch, USE_CUDA, 'train')

				rnet = rnet.train()
				optimizer.zero_grad()
				start_index_log_probabilities, end_index_log_probabilities = \
					rnet(passage, passage_lengths, passagechars, passagechar_lengths,\
						question, question_lengths, questionchars, questionchar_lengths)	

				loss = criterion(start_index_log_probabilities, start_indices)
				loss = loss + criterion(end_index_log_probabilities, end_indices)
				loss.backward()
				optimizer.step()

				best_span = rnet.find_span(start_index_log_probabilities.data.cpu(), end_index_log_probabilities.data.cpu())	
				for index, span in enumerate(best_span):
					instance_id = instance_ids[index]
					current_passage_tokens = passage_tokens[index]
					answer = getanswer(instance_id, current_passage_tokens, span)
					predictiondict[instance_id] = answer
					predictiondictadv[instance_id] = post_process(answer)

			if((i+1)%2 == 0):
					print('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}'.format( \
							epoch+1, num_epochs, i+1, 90000//batch_size, loss.data[0]))
					sys.stdout.flush()

		#train_em, train_f1 = evaluate(train_dataloader, rnet, traingroundtruth)
		predictionadvjson = json.loads(json.dumps(predictiondictadv))
		resultadvjson = evaluator.evaluate(traingroundtruth['data'], predictionadvjson)
		val_em, val_f1 = evaluate(dev_dataloader, rnet, devgroundtruth)
		train_em = resultadvjson['exact_match']
		train_f1 = resultadvjson['f1']
		print('Epoch: [{0}/{1}], Train Acc: [{2} | {3}], Validation Acc:[{4} | {5}]'.format( \
				epoch+1, num_epochs, train_em, train_f1, val_em, val_f1))
		sys.stdout.flush()

		if val_f1 > best_val_f1:
			best_val_f1 = val_f1
			filename = 'rnet' + '.pth'
			file = open(os.path.join(save_dir, filename), 'wb')
			torch.save({'state_dict':rnet.state_dict(), 'dict_args':dict_args}, file)
			print('Saving the model to {}'.format(save_dir))
			file.close()
	return


def train_bidaf():
	cur_dir = os.getcwd()
	glove_dir = '../glove'
	glove_filename = 'glove.6B.100d.txt'
	glove_filepath = os.path.join(glove_dir, glove_filename)

	batch_size = 60

	file_names = ['input/train.json']
	files = [os.path.join(cur_dir,file) for file in file_names]
	train_dataloader, vocab, glove = loader.get_train_data(files, glove_filepath, batch_size)

	file_names = ['input/dev.json']
	files = [os.path.join(cur_dir,file) for file in file_names]
	dev_dataloader = loader.get_test_data(files, vocab, glove, batch_size)

	traingroundtruth = json.load(open(os.path.join(cur_dir,'data/train-v1.1.json')))
	devgroundtruth = json.load(open(os.path.join(cur_dir,'data/dev-v1.1.json')))

	save_dir = 'models/bidafLSTM100100drop0p2new/'

	glovefile = open(os.path.join(save_dir, 'glove.pkl'), 'wb')
	pickle.dump(glove, glovefile)
	glovefile.close()

	vocabfile = open(os.path.join(save_dir, 'vocab.pkl'), 'wb')
	pickle.dump(vocab, vocabfile)
	vocabfile.close()

	dict_args = {
				 'use_charemb': True,
				 'charvocab_size':len(vocab.char2index),
				 'charemb_dim':50,
				 'charemb_rnn_hdim':25,
				 'charemb_rnn_type':'LSTM',
				 'charemb_padix':0,
				 'wordemb_dim':100,
				 'contextemb_rnn_hdim':100,
				 'contextemb_rnn_type':'LSTM',
				 'contextemb_num_layers':3,
				 'bidaf_similarity_function': 'WeightedInputsDotConcatenation',
				 'modelinglayer_rnn_type':'LSTM',
				 'modelinglayer_one_num_layers':2,
				 'modelinglayer_two_num_layers':1,
				 'dropout_rate':0.2
				}
	bidaf = BiDAF(dict_args)

	num_epochs = 20
	learning_rate = 1.0
	criterion = nn.NLLLoss()  
	optimizer = optim.Adadelta(bidaf.parameters(), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=0)
	if USE_CUDA:
		bidaf = bidaf.cuda()
		criterion = criterion.cuda()
	best_val_f1 = 0.0

	for epoch in range(num_epochs):
		predictiondict = {}
		predictiondictadv = {}
		for i,batch in enumerate(train_dataloader):

			passage, passage_lengths, passagechars, passagechar_lengths,\
			question, question_lengths, questionchars, questionchar_lengths,\
			start_indices, end_indices, instance_ids, passage_tokens\
			= convert(batch, USE_CUDA, 'train')

			bidaf = bidaf.train()
			optimizer.zero_grad()
			start_index_log_probabilities, end_index_log_probabilities = \
				bidaf(passage, passage_lengths, passagechars, passagechar_lengths,\
					question, question_lengths, questionchars, questionchar_lengths)

			loss = criterion(start_index_log_probabilities, start_indices)
			loss = loss + criterion(end_index_log_probabilities, end_indices)
			loss.backward()
			optimizer.step()

			if((i+1)%2 == 0):
					print('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}'.format( \
							epoch+1, num_epochs, i+1, 90000//batch_size, loss.data[0]))
					sys.stdout.flush()

			best_span = bidaf.find_span(start_index_log_probabilities.data.cpu(), end_index_log_probabilities.data.cpu())	
			for index, span in enumerate(best_span):
				instance_id = instance_ids[index]
				current_passage_tokens = passage_tokens[index]
				answer = getanswer(instance_id, current_passage_tokens, span)
				predictiondict[instance_id] = answer
				predictiondictadv[instance_id] = post_process(answer)

		#train_em, train_f1 = evaluate(train_dataloader, bidaf, traingroundtruth)
		predictionadvjson = json.loads(json.dumps(predictiondictadv))
		resultadvjson = evaluator.evaluate(traingroundtruth['data'], predictionadvjson)
		val_em, val_f1 = evaluate(dev_dataloader, bidaf, devgroundtruth)
		train_em = resultadvjson['exact_match']
		train_f1 = resultadvjson['f1']
		print('Epoch: [{0}/{1}], Train Acc: [{2} | {3}], Validation Acc:[{4} | {5}]'.format( \
				epoch+1, num_epochs, train_em, train_f1, val_em, val_f1))
		sys.stdout.flush()

		if val_f1 > best_val_f1:
			best_val_f1 = val_f1
			filename = 'bidaf' + '.pth'
			file = open(os.path.join(save_dir, filename), 'wb')
			torch.save({'state_dict':bidaf.state_dict(), 'dict_args':dict_args}, file)
			print('Saving the model to {}'.format(save_dir))
			file.close()
	return

if __name__=='__main__':
	#train_rnet()
	train_bidaf()
