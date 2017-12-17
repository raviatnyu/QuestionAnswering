import os
import torch

from input.vocab import Vocab
from input.config import *

#from vocab import Vocab
#from config import *

class Glove():

	def __init__(self, filepath):
		self.filepath = filepath
		self.word2vec = {}
		self.index2vec = {}

	def create(self, vocab):
		file = open(self.filepath, 'r')
		vector_dim = 0
		for line in file:
			words = line.strip().split(' ')
			word = words[0]
			vec = [float(value) for value in words[1:]]
			vector_dim = len(words[1:])
			if word in vocab.word2index:
				self.word2vec[word] = vec
				self.index2vec[vocab.word2index[word]] = vec
		if WUNK in vocab.word2index:
			random = [float('%.5f'%(torch.rand(1)[0])) for l in range(vector_dim)]
			self.word2vec[WUNK] = random
			self.index2vec[vocab.word2index[WUNK]] = random
		if WPAD in vocab.word2index:
			self.word2vec[WPAD] = [0.0 for l in range(vector_dim)]
			self.index2vec[vocab.word2index[WPAD]] = [0.0 for l in range(vector_dim)]

	def get_word_vectors(self, tokens):
		wvecs = []
		for token in tokens:
			if token.lower() in self.word2vec:
				wvecs.append(self.word2vec[token.lower()])
			else:
				wvecs.append(self.word2vec[WUNK])
		return wvecs

	def get_index_vectors(self, indices):
		wvecs = []
		for index in indices:
			if index in self.index2vec:
				wvecs.append(self.index2vec[index])		
			else:
				#print ("Not found in Vocab replacing with UNK  " + str(index))
				wvecs.append(self.word2vec[WUNK])	
		return wvecs

if __name__ == '__main__':
	glove_dir = './glove'
	glove_filename = 'glove.6B.50d.txt'
	glove_filepath = os.path.join(glove_dir, glove_filename)

	glove = Glove(glove_filepath)
	vocab = Vocab(['train.json'])
	vocab.add_padunk_vocab()
	vocab.create()
	glove.create(vocab)

	print(glove.get_word_vectors(['The','hjfad']))

