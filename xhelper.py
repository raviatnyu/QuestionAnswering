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

from input import loader
import xeval as evaluator
from data import post_process, extra_token_splits

#reload(sys)
#sys.setdefaultencoding('utf-8')

def getanswer(instance_id, current_passage_tokens, span):
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
	return answer

def convert(batch, USE_CUDA, mode):
	if mode == 'train':
		passage = Variable(torch.FloatTensor(batch[0]))
		passage_lengths = torch.LongTensor(batch[1])
		passagechars = Variable(torch.LongTensor(batch[2]))
		passagechar_lengths = torch.LongTensor(batch[3])
		question = Variable(torch.FloatTensor(batch[4]))
		question_lengths = torch.LongTensor(batch[5])
		questionchars = Variable(torch.LongTensor(batch[6]))
		questionchar_lengths = torch.LongTensor(batch[7])
		answer_span = torch.LongTensor(batch[8])
		instance_ids = batch[9]
		passage_tokens = batch[10]
		start_indices, end_indices = torch.chunk(answer_span,2,-1)
		start_indices = Variable(start_indices.squeeze())
		end_indices = Variable(end_indices.squeeze())
	elif mode == 'eval':
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

	if mode == 'compare':
		question_tokens = batch[11]

	if USE_CUDA:
		passage = passage.cuda()
		passage_lengths = passage_lengths.cuda()
		passagechars = passagechars.cuda()
		passagechar_lengths = passagechar_lengths.cuda()
		question = question.cuda()
		question_lengths = question_lengths.cuda()
		questionchars = questionchars.cuda()
		questionchar_lengths = questionchar_lengths.cuda()

	if USE_CUDA and mode == 'train':
		answer_span = answer_span.cuda()
		start_indices = start_indices.cuda()
		end_indices = end_indices.cuda()	

	if mode == 'train':
		return passage, passage_lengths, passagechars, passagechar_lengths,\
			   question, question_lengths, questionchars, questionchar_lengths,\
			   start_indices, end_indices, instance_ids, passage_tokens
	elif mode == 'eval':
		return passage, passage_lengths, passagechars, passagechar_lengths,\
			   question, question_lengths, questionchars, questionchar_lengths,\
			   instance_ids, passage_tokens
	elif mode == 'compare':
		return passage, passage_lengths, passagechars, passagechar_lengths,\
			   question, question_lengths, questionchars, questionchar_lengths,\
			   answer_span, instance_ids, passage_tokens, question_tokens


def get_rnet_subbatch(superbatchlen):
	if len(superbatchlen) > 500:
			subbatchzise = 4
	elif len(superbatchlen) > 450:
			subbatchzise = 6
	elif len(superbatchlen) > 400:
			subbatchzise = 10
	elif len(superbatchlen) > 350:
			subbatchzise = 15
	elif len(superbatchlen) > 250:
			subbatchzise = 15
	elif len(superbatchlen) > 200:
			subbatchzise = 20
	elif len(superbatchlen) > 150:
			subbatchzise = 30
	elif len(superbatchlen) > 100:
			subbatchzise = 30
	else:
			subbatchzise = 60
	return subbatchzise

def get_eval_subbatch(superbatchlen):
	if len(superbatchlen) > 500:
		subbatchzise = 5
	elif len(superbatchlen) > 400:
		subbatchzise = 10
	elif len(superbatchlen) > 300:
		subbatchzise = 15
	elif len(superbatchlen) > 200:
		subbatchzise = 20
	elif len(superbatchlen) > 100:
		subbatchzise = 30
	else:
		subbatchzise = 60
	return subbatchzise




