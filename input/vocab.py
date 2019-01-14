import os
import sys
import json

from collections import Counter

from input.config import *
from input.dataiter import *

class Vocab():

	def __init__(self, files):
		self.files = files
		self.word2index = {}
		self.index2word = {}
		self.char2index = {}
		self.index2char = {}
		
		self.word_counter = Counter()
		self.char_counter = Counter()
		self.num_questions = 0

	def add_padunk_vocab(self):
		self.index2word[len(self.word2index)] = WPAD
		self.word2index[WPAD] = len(self.word2index)
		self.index2char[len(self.char2index)] = CPAD
		self.char2index[CPAD] = len(self.char2index)
		self.index2word[len(self.word2index)] = WUNK
		self.word2index[WUNK] = len(self.word2index)
		self.index2char[len(self.char2index)] = CUNK
		self.char2index[CUNK] = len(self.char2index)
		return

	def update_word_counter(self, tokens):
		if tokens is not None:
			tokens = [token.lower() for token in tokens]
			self.word_counter.update(tokens)
		return

	def update_char_counter(self, tokens):
		if tokens is not None:
			for token in tokens:
				self.char_counter.update(token)
		return

	def create_vocab_map_top_k(self, counter, topk, item2index, index2item):
		topkcounter = counter.most_common(topk)
		cur_length = len(item2index)
		for index in range(0,len(topkcounter)):
			item2index[topkcounter[index][0]] = cur_length + index
			index2item[cur_length + index] = topkcounter[index][0]
		return

	def create_vocab_map_min_frequency(self, counter, minf, item2index, index2item):
		allcounter = counter.most_common()
		cur_length = len(item2index)
		for index in range(0,len(allcounter)):
			if allcounter[index][1] < minf:
				break
			item2index[allcounter[index][0]] = cur_length + index
			index2item[cur_length + index] = allcounter[index][0]
		return

	def create_vocab_map_all(self, counter, item2index, index2item):
		allcounter = list(counter.items())
		cur_length = len(item2index)
		for index in range(0,len(allcounter)):
			item2index[allcounter[index][0]] = cur_length + index
			index2item[cur_length + index] = allcounter[index][0]
		return

	def create_vocab_map(self):
		if Word_Vocab_Filter == Top_K:
			self.create_vocab_map_top_k(self.word_counter, Word_Top_K, self.word2index, self.index2word)
		elif Word_Vocab_Filter == Min_Frequency:
			self.create_vocab_map_min_frequency(self.word_counter, Word_Min_Frequency, self.word2index, self.index2word)
		else:
			self.create_vocab_map_all(self.word_counter, self.word2index, self.index2word)
		if Char_Vocab_Filter == Top_K:
			self.create_vocab_map_top_k(self.char_counter, Char_Top_K, self.char2index, self.index2char)
		elif Char_Vocab_Filter == Min_Frequency:
			self.create_vocab_map_min_frequency(self.char_counter, Char_Min_Frequency, self.char2index, self.index2char)
		else:
			self.create_vocab_map_all(self.char_counter, self.char2index, self.index2char)	
		return

	def create(self):
		for filepath in self.files:
			file = open(filepath, 'r') 
			data = json.load(file)
			for passage_tokens, question_tokens, answer_span, instance_id in squaditershallow(data):
				self.update_word_counter(passage_tokens)
				self.update_word_counter(question_tokens)
				self.update_char_counter(passage_tokens)
				self.update_char_counter(question_tokens)
				if question_tokens is not None:
					self.num_questions = self.num_questions + 1
		self.create_vocab_map()
		return

	def get_word_indices(self, tokens):
		indices = []
		for token in tokens:
			if token.lower() in self.word2index:
				indices.append(self.word2index[token.lower()])
			else:
				indices.append(self.word2index[WUNK])
		return indices

	def get_index_words(self, indices):
		words = []
		for index in indices:
			if index in self.index2word:
				words.append(self.index2word[index])
			else:
				raise
		return words

	def get_char_index(self, token):
		indices = []
		for char in token:
			if char in self.char2index:
				indices.append(self.char2index[char])
			else:
				indices.append(self.char2index[CUNK])
		return indices

	def get_char_indices(self, tokens):
		indices = []
		for token in tokens:
			indices.append(self.get_char_index(token))
		return indices

if __name__ == '__main__':
	vocab = Vocab(['train.json'])
	vocab.add_padunk_vocab()
	vocab.create()
	#print(vocab.get_word_indices(['THe','i','National']))
	#print(vocab.get_char_indices(['THe','das','National']))
	#print(vocab.get_word_indices(['the','i','national']))
	#print(vocab.get_char_indices(['the','das','national']))
	print (vocab.word_counter)
	print (vocab.char_counter)


