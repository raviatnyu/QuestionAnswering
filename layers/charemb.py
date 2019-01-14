import torch
import torch.nn as nn
from torch.autograd import Variable

import layers.utils as utils
#import utils

class RNNCharEmb(nn.Module):
	
	def __init__(self, dict_args):
		super(RNNCharEmb, self).__init__()
		self.cvocab_size = dict_args['cvocab_size']
		self.cembed_dim = dict_args['charemb_dim']
		self.hidden_dim = dict_args['charemb_rnn_hdim']
		self.rnn_type = dict_args['charemb_rnn_type']
		self.padding_idx = dict_args['charemb_padix']
		self.num_layers = 1
		self.use_birnn = True
		if self.use_birnn == True:
			self.num_directions = 2
		else:
			self.num_directions = 1
		self.dropout_rate = dict_args['dropout_rate']


		#embedding layer
		self.char_embed = nn.Embedding(self.cvocab_size, self.cembed_dim, padding_idx=self.padding_idx)
		#hidden layer
		if self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(self.cembed_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.use_birnn, dropout=self.dropout_rate)
			#self.rnn = nn.LSTM(self.cembed_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.use_birnn)		
		elif self.rnn_type == 'GRU':
			self.rnn = nn.GRU(self.cembed_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.use_birnn, dropout=self.dropout_rate)
			#self.rnn = nn.GRU(self.cembed_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.use_birnn)
		elif self.rnn_type == 'RNN':
			pass
		#dropout layer
		if self.dropout_rate > 0: self.dropout_layer = nn.Dropout(p=self.dropout_rate)

	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			h_0 = Variable(weight.new(self.num_layers*self.num_directions, batch_size, self.hidden_dim).zero_())
			c_0 = Variable(weight.new(self.num_layers*self.num_directions, batch_size, self.hidden_dim).zero_())
			return (h_0,c_0)
		elif self.rnn_type == 'GRU':
			h_0 = Variable(weight.new(self.num_layers*self.num_directions, batch_size, self.hidden_dim).zero_())
			return h_0
		elif self.rnn_type == 'RNN':
			pass

	def forward(self, inputs, lengths):
		#input: batch_size*num_words*num_chars
		#lengths: batch_size*num_words
		batch_size = inputs.data.shape[0]
		num_words = inputs.data.shape[1]
		num_chars = inputs.data.shape[2]

		inputs = inputs.permute(1, 2, 0) #inputs: num_words*num_chars*batch_size
		lengths = lengths.permute(1,0) #lengths: num_words*batch_size
		wcembeds = Variable(inputs.data.new(num_words, batch_size, 2*self.hidden_dim).zero_().float()) #wcembeds: num_words*batch_size*2.hidden_dim

		for index, word in enumerate(inputs): #word: num_chars*batch_size
			word = word.permute(1, 0) #word: batch_size*num_chars
			length_seq = lengths[index] #length: batch_size
			word_sorted, sorted_lengths, original_indices = utils.sort_batch(word, length_seq) #word_sorted: batch_size*num_chars
			
			cembeds = self.char_embed(word_sorted) #cembeds: batch_size*num_chars*cembed_dim
			if self.dropout_rate > 0: cembeds = self.dropout_layer(cembeds) #cembeds: batch_size*num_chars*cembed_dim
			cembeds = cembeds.permute(1, 0, 2) #cembeds: num_chars*batch_size*cembed_dim
			cembeds = nn.utils.rnn.pack_padded_sequence(cembeds, list(sorted_lengths)) #cembeds: num_chars*batch_size*cembed_dim

			hidden = self.init_hidden(batch_size)
			if self.rnn_type == 'LSTM':
				rnn_out, (h_n,c_n) = self.rnn(cembeds, hidden) #h_n: 1.2*batch_size*hidden_dim
			elif self.rnn_type == 'GRU':
				rnn_out, h_n = self.rnn(cembeds, hidden) #h_n: 1.2*batch_size*hidden_dim

			forward_hidden = h_n[2*self.num_layers - 2] #forward_hidden: batch_size*hidden_dim
			backward_hidden = h_n[2*self.num_layers - 1] #backward_hidden: batch_size*hidden_dim
			#rnn_out: num_chars*batch_size*num_directions.hidden_dim
			#backward_hidden = rnn_out[0][:,self.hidden_dim:] #backward_hidden: batch_size*hidden_dim
			#forward_hidden = rnn_out[num_chars-1][:,0:self.hidden_dim] #forward_hidden: batch_size*hidden_dim
			
			wcembeds[index] = torch.cat((forward_hidden,backward_hidden),dim=1) #wcembeds[index]: batch_size*2.hidden_dim
			wcembeds[index] = utils.unsort_batch(wcembeds[index], original_indices)
		return wcembeds.permute(1, 0, 2) #wcembeds: batch_size*num_words*2.hidden_dim

if __name__=='__main__':
	dict_args = {
				 'cvocab_size':20, #vocabulary size
				 'charemb_dim':3, #word embedding dimension
				 'charemb_rnn_hdim':3,  #size of the hidden dimension
				 'charemb_rnn_type':'LSTM', #RNN, LSTM, GRU
				 'charemb_padix':0 #padding idx of char vocab
				}
	net = RNNCharEmb(dict_args)
	#inputs = Variable(torch.LongTensor([[1,3,4,7],[11,2,18,4],[3,16,13,9]]))
	inputs = Variable(torch.LongTensor([[[1,3,0,0],[5,6,7,0],[4,3,12,0]],[[11,2,18,8],[12,14,15,10],[0,0,0,0]]]))
	lengths = torch.LongTensor([[2,3,3],[4,4,1]])
	print (net(inputs, lengths))






		    		
