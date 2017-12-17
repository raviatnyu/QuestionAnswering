import os
import sys

import torch
import numpy as numpy
import torch.utils.data as data

from input.vocab import Vocab
from input.glove import Glove
from input.config import *
from input.dataiter import *

#from vocab import Vocab
#from glove import Glove
#from config import *
#from dataiter import *

class Dataset(data.Dataset):
	def __init__(self, files):
		self.files = files
		self.data = [] #list of [passagewords, passagechars, querywords, querychars, span, idx]
		self.lengths = []
		self.size = 0
		self.wpad = 0
		self.cpad = 0

		self.glove = None

		self.maxcharlen = MAX_CHAR_LEN
		self.maxpassagelen = MAX_PASSAGE_LEN
		self.maxquerylen = MAX_QUERY_LEN

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		return self.data[index]

	def set_pad_indices(self, vocab):
		self.wpad = vocab.word2index[WPAD]
		self.cpad = vocab.char2index[CPAD]

	def add_glove_vecs(self, glove):
		self.glove = glove

	def create(self, vocab):
		def get_insert_position(current_length):
			for i,length in enumerate(self.lengths):
				if length < current_length:
					return i
			return len(self.lengths)

		for filepath in self.files:
			file = open(filepath, 'r') 
			data = json.load(file)
			cur_passage_tokens = []
			cur_question_tokens = []
			cur_instance_id = ''
			for passage_tokens, question_tokens, answer_span, instance_id in squaditershallow(data):
				if passage_tokens is not None:
					cur_passage_tokens = passage_tokens
				if question_tokens is not None:
					cur_question_tokens = question_tokens
					cur_instance_id = instance_id
				passagewords = vocab.get_word_indices(cur_passage_tokens)
				passagechars = vocab.get_char_indices(cur_passage_tokens)
				querywords = vocab.get_word_indices(cur_question_tokens)
				querychars = vocab.get_char_indices(cur_question_tokens)

				current_length = len(cur_passage_tokens)
				insert_position = get_insert_position(current_length)
				if USE_CHAR_EMB:
					self.data.insert(insert_position, [passagewords,passagechars,querywords,querychars,answer_span,instance_id,cur_passage_tokens,cur_question_tokens])
					self.lengths.insert(insert_position, current_length)
				else:
					self.data.insert(insert_position, [passagewords,[],querywords,[],answer_span,instance_id,cur_passage_tokens,cur_question_tokens])
					self.lengths.insert(insert_position, current_length)				
		self.size = len(self.data)

	def collate_fn(self, mini_batch):
		def get_padded_list_normal(sequence_list, padding_value):
			sequence_lengths = [len(sequence) for sequence in sequence_list]
			max_length = max(sequence_lengths)
			num_sequences = len(sequence_list)
			padded_sequence_list = [[padding_value for col in range(max_length)] for row in range(num_sequences)]
			for index, sequence in enumerate(sequence_list):
				padded_sequence_list[index][:sequence_lengths[index]] = sequence
			return padded_sequence_list, sequence_lengths

		def get_padded_list_truncated(sequence_list, padding_value, max_length):
			sequence_lengths = [len(sequence) for sequence in sequence_list]
			num_sequences = len(sequence_list)
			padded_sequence_list = [[padding_value for col in range(max_length)] for row in range(num_sequences)]
			for index, sequence in enumerate(sequence_list):
				if sequence_lengths[index] < max_length:
					padded_sequence_list[index][:sequence_lengths[index]] = sequence
				else:
					padded_sequence_list[index][:max_length] = sequence[:max_length]
					sequence_lengths[index] = max_length
			return padded_sequence_list, sequence_lengths

		def get_padded_indexes_list(sequence_list, padding_value):
			padded_sequence_list, sequence_lengths = get_padded_list_normal(sequence_list, [padding_value]) #assumes pad char to make seqlen > 0
			sequence_length = len(padded_sequence_list[0])
			num_sequences = len(padded_sequence_list)
			padded_chars_batch = [[] for i in range(num_sequences)]
			char_sequence_lengths = [[] for i in range(num_sequences)]
			for col in range(sequence_length):
				ithword_chars_batch = [padded_sequence_list[row][col] for row in range(num_sequences)]
				padded_ithword_chars_batch, ithword_sequence_lengths = get_padded_list_truncated(ithword_chars_batch, padding_value, self.maxcharlen)
				for row in range(num_sequences):
					padded_chars_batch[row].append(padded_ithword_chars_batch[row])
					char_sequence_lengths[row].append(ithword_sequence_lengths[row])
			return padded_chars_batch, char_sequence_lengths

		def replace_indices_with_vecs(sequence_list):
			return [self.glove.get_index_vectors(sequence) for sequence in sequence_list]

		passagewords_batch,passagechars_batch,querywords_batch,\
		querychars_batch,answer_span_batch,instance_id_batch,passage_tokens_batch,question_tokens_batch = zip(*mini_batch)
		padded_passagewords_batch, passage_sequence_lengths = get_padded_list_normal(passagewords_batch, self.wpad)
		padded_querywords_batch, query_sequence_lengths = get_padded_list_normal(querywords_batch, self.wpad)
		padded_passagechars_batch, passagechar_sequence_lengths = get_padded_indexes_list(passagechars_batch, self.cpad)
		padded_querychars_batch, querychar_sequence_lengths = get_padded_indexes_list(querychars_batch, self.cpad)
		padded_passagewordvecs_batch = replace_indices_with_vecs(padded_passagewords_batch)
		padded_querywordvecs_batch = replace_indices_with_vecs(padded_querywords_batch)

		if USE_CHAR_EMB:
			return padded_passagewordvecs_batch, passage_sequence_lengths, \
			padded_passagechars_batch, passagechar_sequence_lengths, \
			padded_querywordvecs_batch, query_sequence_lengths, \
			padded_querychars_batch, querychar_sequence_lengths, \
			answer_span_batch, instance_id_batch, passage_tokens_batch, question_tokens_batch 
		else:
			return padded_passagewordvecs_batch, passage_sequence_lengths, \
			[], [], \
			padded_querywordvecs_batch, query_sequence_lengths, \
			[], [], \
			answer_span_batch, instance_id_batch, passage_tokens_batch, question_tokens_batch

if __name__=='__main__':
	dataset = Dataset(['dev.json'])
	vocab = Vocab(['train.json'])
	vocab.add_padunk_vocab()
	vocab.create()
	dataset.set_pad_indices(vocab)
	dataset.create(vocab)
	glove_dir = './glove'
	glove_filename = 'glove.6B.50d.txt'
	glove_filepath = os.path.join(glove_dir, glove_filename)
	glove = Glove(glove_filepath)
	glove.create(vocab)
	dataset.add_glove_vecs(glove)
	print(dataset.collate_fn([dataset.__getitem__(0), dataset.__getitem__(234)])[6])
	print(dataset.collate_fn([dataset.__getitem__(0), dataset.__getitem__(234)])[7])

