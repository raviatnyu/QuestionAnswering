import torch
import torch.nn as nn
from torch.autograd import Variable

import layers.utils as utils
import layers.similarity as similarity
from layers.unidirattention import UniDirAttention
from layers.unidirattention import UniDirAttentionItr
from layers.gate import Gate

class MatchAttention(nn.Module):

	def __init__(self, dict_args):
		super(MatchAttention, self).__init__()
		self.sequence1_dim = dict_args['sequence1_dim']
		self.sequence2_dim = dict_args['sequence2_dim']
		self.hidden_dim = dict_args['rnn_hdim']
		self.input_dim = self.sequence1_dim + self.sequence2_dim
		self.rnn_type = dict_args['rnn_type']
		self.gated_attention = dict_args['gated_attention']
		self.similarity_function = dict_args['similarity_function']

		if self.gated_attention:
			self.gate = Gate({'sigmoidinputdim':self.input_dim,'gateinputdim':self.input_dim})
		if self.rnn_type == 'LSTM':
			self.rnn = nn.LSTMCell(self.input_dim, self.hidden_dim)
		elif self.rnn_type == 'GRU':
			self.rnn = nn.GRUCell(self.input_dim, self.hidden_dim)
		elif self.rnn_type == 'RNN':
			pass

		if dict_args['similarity_function'] == 'ProjectionSimilaritySharedWeights':
			self.attention_function = UniDirAttention({'similarity_function': dict_args['similarity_function'], \
				'similarity_function_pointer': dict_args['similarity_function_pointer']})
		else:
			self.attention_function = UniDirAttention({'similarity_function': dict_args['similarity_function'], 'sequence1_dim':self.sequence2_dim, \
				'sequence2_dim':self.sequence1_dim+self.hidden_dim, 'projection_dim':dict_args['projection_dim']})

	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			h_0 = Variable(weight.new(batch_size, self.hidden_dim).zero_())
			c_0 = Variable(weight.new(batch_size, self.hidden_dim).zero_())
			return (h_0,c_0)
		elif self.rnn_type == 'GRU':
			h_0 = Variable(weight.new(batch_size, self.hidden_dim).zero_())
			return (h_0,None)
		elif self.rnn_type == 'RNN':
			pass

	def forward(self, sequence1, sequence2, sequence1_mask, sequence2_mask, reverse=False):
		#sequence1: batch_size*num_words1*iembed1
		#sequence2: batch_size*num_words2*iembed2

		sequence1 = sequence1.permute(1,0,2) #sequence1: num_words1*batch_size*iembed1
		sequence1_mask = sequence1_mask.permute(1,0) #sequence1_mask: num_words1*batch_size
		num_words1, batch_size, _ = sequence1.size()
		_, num_words2, _ = sequence2.size()

		sequence1_sequence2_matchattn = Variable(sequence1.data.new(num_words1,batch_size,self.hidden_dim).zero_())
		sequence2_matchattn_weights = None
		if not self.training: sequence2_matchattn_weights = Variable(sequence1.data.new(num_words1,batch_size,num_words2).zero_())
		#sequence1_sequence2_matchattn: num_words1*batch_size*hidden_dim
		#sequence2_matchattn_weights: num_words1*batch_size*num_words2

		h_t, c_t = self.init_hidden(batch_size)
		for ith_item in range(num_words1):
			hidden_mask = sequence1_mask[ith_item]
			if self.similarity_function == 'ProjectionSimilaritySharedWeights':
				vector_sequence2_attention, sequence2_attention_weights = self.attention_function(sequence2, sequence1[ith_item], sequence2_mask, vector2=h_t)				
			else:
				vector = torch.cat((sequence1[ith_item], h_t), dim=1)#vector: batch_size*(iembed1+hidden_dim)
				vector_sequence2_attention, sequence2_attention_weights = self.attention_function(sequence2, vector, sequence2_mask)
			#vector_sequence2_attention: batch_size*iembed2
			#sequence2_attention_weights: batch_size*num_words2

			input = torch.cat((sequence1[ith_item], vector_sequence2_attention), dim=1) #input: batch_size*(iembed1+iembed2)
			if self.gated_attention:
				#Implement Gated Attention create class 
				input = self.gate(input, input)
			if self.rnn_type == 'LSTM':
				h_t, c_t = self.rnn(input, (h_t, c_t)) #h_t: batch_size*hidden_dim
				if reverse:
					h_t = utils.mask_sequence(h_t, hidden_mask)
					c_t = utils.mask_sequence(c_t, hidden_mask)
			elif self.rnn_type == 'GRU':
				h_t = self.rnn(input, h_t) #h_t: batch_size*hidden_dim
				if reverse:
					h_t = utils.mask_sequence(h_t, hidden_mask)
			elif self.rnn_type == 'RNN':
				pass
			sequence1_sequence2_matchattn[ith_item] = h_t
			if not self.training: sequence2_matchattn_weights[ith_item] = sequence2_attention_weights

		sequence1_sequence2_matchattn = sequence1_sequence2_matchattn.permute(1,0,2)
		if not self.training: sequence2_matchattn_weights = sequence2_matchattn_weights.permute(1,0,2)
		sequence1_mask = sequence1_mask.permute(1,0) #sequence1_mask: batch_size*num_words1
		sequence1_sequence2_matchattn = utils.mask_sequence(sequence1_sequence2_matchattn, sequence1_mask)
		if not self.training: sequence2_matchattn_weights = utils.mask_sequence(sequence2_matchattn_weights, sequence1_mask)
		#sequence1_sequence2_matchattn: batch_size*num_words1*hidden_dim
		#sequence2_matchattn_weights: batch_size*num_words1*num_words2
		return sequence1_sequence2_matchattn, sequence2_matchattn_weights

if __name__=='__main__':
	matchattn = MatchAttention({'similarity_function': 'WeightedSumProjection', 'sequence1_dim':12, 'sequence2_dim':8, 'projection_dim':10, 'rnn_type':'LSTM', 'rnn_hdim':15, 'gated_attention':True})
	sequence1_sequence2_matchattn, sequence2_matchattn_weights =\
		matchattn(Variable(torch.randn(3,6,12)), Variable(torch.randn(3,4,8)),\
		Variable(utils.sequence_mask(torch.LongTensor([5,3,6]))), Variable(utils.sequence_mask(torch.LongTensor([4,3,2]))), reverse=True)
	print(sequence1_sequence2_matchattn)
	print(sequence2_matchattn_weights)

