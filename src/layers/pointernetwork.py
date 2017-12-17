import torch
import torch.nn as nn
from torch.autograd import Variable

import layers.utils as utils
import layers.similarity as similarity
from layers.unidirattention import UniDirAttention
#import utils as utils
#import similarity as similarity
#from unidirattention import UniDirAttention

class PointerNetwork(nn.Module):

	def __init__(self, dict_args):
		super(PointerNetwork, self).__init__()
		self.sequence_dim = dict_args['sequence_dim']
		self.hidden_dim = dict_args['rnn_hdim']
		self.input_dim = self.sequence_dim
		self.rnn_type = dict_args['rnn_type']

		if self.rnn_type == 'LSTM':
			self.rnn = nn.LSTMCell(self.input_dim, self.hidden_dim) #ToDO
		elif self.rnn_type == 'GRU':
			self.rnn = nn.GRUCell(self.input_dim, self.hidden_dim)
		elif self.rnn_type == 'RNN':
			pass

		if dict_args['similarity_function'] == 'ProjectionSimilaritySharedWeights':
			self.attention_function = UniDirAttention({'similarity_function': dict_args['similarity_function'], \
				'similarity_function_pointer': dict_args['similarity_function_pointer']})
		else:
			self.attention_function = UniDirAttention({'similarity_function': dict_args['similarity_function'], 'sequence1_dim':self.sequence_dim, \
				'sequence2_dim':self.hidden_dim, 'projection_dim':dict_args['projection_dim']})

	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			c_0 = Variable(weight.new(batch_size, self.hidden_dim).zero_())
			return c_0
		elif self.rnn_type == 'GRU':
			pass
		elif self.rnn_type == 'RNN':
			pass

	def forward(self, sequence, hidden_t, sequence_mask, num_steps):
		#sequence: batch_size*num_words*iembed
		#hidden_t: batch_size*hidden_dim

		batch_size, num_words, _ = sequence.size()

		h_t = hidden_t
		#c_t = self.init_hidden(batch_size)
		pointer_sequence_attention_weights = Variable(sequence.data.new(num_steps, batch_size, num_words).zero_())
		#pointer_sequence_attention_weights: num_steps*batch_size*num_words

		for step in range(num_steps):
			vector_sequence_attention, sequence_attention_weights = self.attention_function(sequence, h_t, sequence_mask, softmax=False)
			if self.rnn_type == 'LSTM':
				h_t, c_t = self.rnn(vector_sequence_attention, (h_t, c_t)) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'GRU':
				h_t = self.rnn(vector_sequence_attention, h_t) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'RNN':
				pass
			pointer_sequence_attention_weights[step] = sequence_attention_weights

		pointer_sequence_attention_weights = pointer_sequence_attention_weights.permute(1,0,2)

		return pointer_sequence_attention_weights #batch_size*num_steps*num_words

if __name__=='__main__':
	'''ptrnet = PointerNetwork({'similarity_function': 'WeightedSumProjection', 'sequence_dim':15, 'projection_dim':10, 'rnn_type':'GRU', 'rnn_hdim':15})
	pointer_sequence_attention_weights =\
		ptrnet(Variable(torch.randn(3,6,15)), Variable(torch.randn(3,15)),\
		Variable(utils.sequence_mask(torch.LongTensor([5,3,6]))), 2)
	print(pointer_sequence_attention_weights)'''

	ptrnet = PointerNetwork({'similarity_function': 'WeightedSumProjection', 'sequence_dim':15, 'projection_dim':10, 'rnn_type':'GRU', 'rnn_hdim':15})
	pointer_sequence_attention_weights =\
		ptrnet(Variable(torch.randn(3,6,15)), Variable(torch.randn(1,15).expand(3,15)),\
		Variable(utils.sequence_mask(torch.LongTensor([5,3,6]))), 2)
	print(pointer_sequence_attention_weights)

