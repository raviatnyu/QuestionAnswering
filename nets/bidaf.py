import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional

from layers import RNNCharEmb
from layers import BiDirEncoder
from layers import BiDirAttention
from layers import utils as utils
from layers import similarity as similarity

class BiDAF(nn.Module):

	def __init__(self, dict_args):
		super(BiDAF, self).__init__()

		#character embedding layer
		self.use_charemb = dict_args['use_charemb']
		self.charemb_rnn_hdim = 0
		if self.use_charemb:
			self.cvocab_size = dict_args['charvocab_size']
			self.charemb_dim = dict_args['charemb_dim']
			self.charemb_rnn_hdim = dict_args['charemb_rnn_hdim']
			self.charemb_rnn_type = dict_args['charemb_rnn_type']
			self.charemb_padix = dict_args['charemb_padix']

		#input embedding layer
		self.wordemb_dim = dict_args['wordemb_dim']
		self.contextemb_rnn_hdim = dict_args['contextemb_rnn_hdim']
		self.contextemb_rnn_type = dict_args['contextemb_rnn_type']
		self.contextemb_num_layers = dict_args['contextemb_num_layers']

		#bidaf layer
		self.bidaf_similarity_function = dict_args['bidaf_similarity_function']

		#modeling layer
		self.modelinglayer_rnn_hdim = dict_args['contextemb_rnn_hdim']
		self.modelinglayer_rnn_type = dict_args['modelinglayer_rnn_type']
		self.modelinglayer_one_num_layers = dict_args['modelinglayer_one_num_layers']
		self.modelinglayer_two_num_layers = dict_args['modelinglayer_two_num_layers']

		#dropout layer
		self.use_dropout = False
		self.dropout_rate = 0
		if dict_args['dropout_rate'] > 0:
			self.use_dropout = True
			self.dropout_rate = dict_args['dropout_rate']

		#######Dropout layer
		if self.use_dropout: self.dropout_layer = nn.Dropout(p=self.dropout_rate)

		#######character embedding layer
		if self.use_charemb:
			charemb_layer_args = {
									'cvocab_size': self.cvocab_size, 
									'charemb_dim': self.charemb_dim, 
									'charemb_rnn_hdim': self.charemb_rnn_hdim,  
									'charemb_rnn_type': self.charemb_rnn_type,
									'charemb_padix': self.charemb_padix,
									'dropout_rate' : self.dropout_rate 
								 }
			self.charemb_layer = RNNCharEmb(charemb_layer_args)

		#context embedding layer
		contextemb_layer_args = {
								   'input_dim': 2*self.charemb_rnn_hdim + self.wordemb_dim , 
								   'rnn_hdim': self.contextemb_rnn_hdim,  
								   'rnn_type': self.contextemb_rnn_type, 
								   'num_layers': self.contextemb_num_layers,
								   'dropout_rate' : self.dropout_rate 
								}
		self.contextemb_layer = BiDirEncoder(contextemb_layer_args)

		#bidaf layer
		bidaf_layer_args = {
							  'similarity_function': self.bidaf_similarity_function,
							  'sequence1_dim': 2*self.contextemb_rnn_hdim,	
							  'sequence2_dim': 2*self.contextemb_rnn_hdim,
							  'similarity_input_dim': 2*self.contextemb_rnn_hdim,
							  'one_shot_attention':True,
							  'self_match_attention':False
						   }
		self.bidaf_layer = BiDirAttention(bidaf_layer_args)

		#modeling layer one
		modeling_layer_one_args = {
									'input_dim': 8*self.contextemb_rnn_hdim, 
									'rnn_hdim': self.modelinglayer_rnn_hdim,  
									'rnn_type': self.modelinglayer_rnn_type, 
									'num_layers': self.modelinglayer_one_num_layers,
									'dropout_rate' : 0 
							  	  }
		self.modeling_layer_one = BiDirEncoder(modeling_layer_one_args)
		#modeling layer two
		modeling_layer_two_args = {
									'input_dim': 2*self.modelinglayer_rnn_hdim, 
									'rnn_hdim': self.modelinglayer_rnn_hdim,  
									'rnn_type': self.modelinglayer_rnn_type, 
									'num_layers': self.modelinglayer_two_num_layers,
									'dropout_rate' : 0 
							  	  }
		self.modeling_layer_two = BiDirEncoder(modeling_layer_two_args)

		#output layer
		self.start_index_linear = nn.Linear(8*self.contextemb_rnn_hdim + 2*self.modelinglayer_rnn_hdim, 1)
		self.end_index_linear = nn.Linear(8*self.contextemb_rnn_hdim + 2*self.modelinglayer_rnn_hdim, 1)

	def forward(self, passage, passage_lengths, passagechars, passagechar_lengths,\
				question, question_lengths, questionchars, questionchar_lengths):
		#passage: batch_size*num_words_passage*wembdim
		#passage_lengths: batch_size
		#passagechars: batch_size*num_words_passage*num_max_chars
		#passagechar_lengths: batch_size*num_words_passage

		#question: batch_size*num_words_question*wembdim
		#question_lengths: batch_size
		#questionchars: batch_size*num_words_question*num_max_chars
		#questionchar_lengths: batch_size*num_words_question

		passage_embedding = passage[:,0:passage_lengths.max()]
		question_embedding = question[:,0:question_lengths.max()]
		passage_mask = Variable(utils.sequence_mask(passage_lengths)) #passage_mask: batch_size*num_words_passage
		question_mask = Variable(utils.sequence_mask(question_lengths)) #question_mask: batch_size*num_words_question		

		##### Character Embedding Layer
		if self.use_charemb:
			passage_char_embedding = self.charemb_layer(passagechars, passagechar_lengths)
			#passage_char_embedding: batch_size*num_words_passage*2.charemb_rnn_hdim
			question_char_embedding = self.charemb_layer(questionchars, questionchar_lengths)
			#question_char_embedding: batch_size*num_words_question*2.charemb_rnn_hdim

			##### Char and Word Embedding Concatenation
			passage_embedding = torch.cat((passage, passage_char_embedding), dim = 2)
			question_embedding = torch.cat((question, question_char_embedding), dim = 2)
			passage_embedding = passage_embedding[:,0:passage_lengths.max()]
			question_embedding = question_embedding[:,0:question_lengths.max()]
			passage_embedding = utils.mask_sequence(passage_embedding, passage_mask)
			question_embedding = utils.mask_sequence(question_embedding, question_mask)
			#passage_embedding: batch_size*num_words_passage*(2.charemb_rnn_hdim+wembdim)
			#question_embedding: batch_size*num_words_question*(2.charemb_rnn_hdim+wembdim)

		##### Context Embedding Layer
		#passage_embedding, passage_lengths = self.contextemb_layer(passage_embedding, passage_lengths)
		#question_embedding, question_lengths = self.contextemb_layer(question_embedding, question_lengths)
		passage_embedding, _ = self.contextemb_layer(passage_embedding, passage_lengths)
		question_embedding, _ = self.contextemb_layer(question_embedding, question_lengths)
		#passage_embedding: batch_size*num_words_passage*2.contextemb_rnn_hdim
		#question_embedding: batch_size*num_words_question*2.contextemb_rnn_hdim
		#Skipping recomputation of passage_mask and question_mask

		##### BiDAF Layer
		passage_question_attention, question_attention_weights, question_passage_attention, passage_attention_weights =\
			self.bidaf_layer(passage_embedding, question_embedding, passage_mask, question_mask)
		#passage_question_attention: batch_size*num_words_passage*2.contextemb_rnn_hdim 
		#question_attention_weights: batch_size*num_words_passage*num_words_question
		#question_passage_attention: batch_size*2.contextemb_rnn_hdim 
		#passage_attention_weights: batch_size*num_words_passage

		question_passage_attention = question_passage_attention.unsqueeze(1) #question_passage_attention: batch_size*1*2.contextemb_rnn_hdim 
		question_passage_attention = question_passage_attention.expand(question_passage_attention.size(0), passage_question_attention.size(1), question_passage_attention.size(2))
		#question_passage_attention: batch_size*num_words_passage*2.contextemb_rnn_hdim 
		#passage_question_attention: batch_size*num_words_passage*2.contextemb_rnn_hdim 

		#Skipping masking for passage_question_attention, question_passage_attention 

		question_aware_passage_representation = torch.cat((passage_embedding,\
														   passage_question_attention,\
														   passage_embedding*passage_question_attention,\
														   passage_embedding*question_passage_attention), dim = -1)
		#question_aware_passage_representation: batch_size*num_words_passage*8.contextemb_rnn_hdim

		if self.use_dropout: question_aware_passage_representation = self.dropout_layer(question_aware_passage_representation)

		question_aware_passage_context_representation_one, _ = self.modeling_layer_one(question_aware_passage_representation, passage_lengths)
		#question_aware_passage_context_representation_one: batch_size*num_words_passage*2.modelinglayer_rnn_hdim
		if self.use_dropout: question_aware_passage_context_representation_one = self.dropout_layer(question_aware_passage_context_representation_one)
		
		question_aware_passage_context_representation_two, _ = self.modeling_layer_two(question_aware_passage_context_representation_one, passage_lengths)
		#question_aware_passage_context_representation_two: batch_size*num_words_passage*2.modelinglayer_rnn_hdim
		if self.use_dropout: question_aware_passage_context_representation_two = self.dropout_layer(question_aware_passage_context_representation_two)


		start_index_representation = torch.cat((question_aware_passage_representation,\
												question_aware_passage_context_representation_one), dim = -1)
		end_index_representation = torch.cat((question_aware_passage_representation,\
											  question_aware_passage_context_representation_two), dim = -1)
		#start_index_representation: batch_size*num_words_passage*(8.contextemb_rnn_hdim + 2.modelinglayer_rnn_hdim)
		#end_index_representation: batch_size*num_words_passage*(8.contextemb_rnn_hdim + 2.modelinglayer_rnn_hdim)


		start_index_values = self.start_index_linear(start_index_representation).squeeze()
		end_index_values = self.end_index_linear(end_index_representation).squeeze()

		'''#start_index_probabilities: batch_size*num_words_passage
		#end_index_probabilities: batch_size*num_words_passage
		start_index_probabilities = utils.masked_softmax(start_index_values, passage_mask.float())
		end_index_probabilities = utils.masked_softmax(end_index_values, passage_mask.float())'''

		start_index_log_probabilities = start_index_values + passage_mask.float().log()
		end_index_log_probabilities = end_index_values + passage_mask.float().log()
		start_index_log_probabilities = functional.log_softmax(start_index_log_probabilities)
		end_index_log_probabilities = functional.log_softmax(end_index_log_probabilities)
		#start_index_log_probabilities: batch_size*num_words_passage
		#end_index_log_probabilities: batch_size*num_words_passage

		#uncomment the below line in compare mode
		#return start_index_log_probabilities, end_index_log_probabilities, question_attention_weights
		return start_index_log_probabilities, end_index_log_probabilities

	#accepts cpu tensors after converting variables to .data.cpu()
	def find_span(self, start_index_log_probabilities, end_index_log_probabilities):
		#start_index_log_probabilities: batch_size*num_words_passage
		#end_index_log_probabilities: batch_size*num_words_passage
		batch_size, num_classes = start_index_log_probabilities.size()
		best_span = torch.zeros(batch_size, 2)

		for batch in range(batch_size):
			start_probs = start_index_log_probabilities[batch]
			end_probs = end_index_log_probabilities[batch]

			start_prob_max_value = -1e20
			start_prob_max_index = -1
			span_prob_max_value = -1e20
			span_prob_max_indices = [-1,-1]

			for i in range(num_classes):
				if (start_probs[i] > start_prob_max_value):
					start_prob_max_value = start_probs[i]
					start_prob_max_index = i

				if ((end_probs[i] + start_prob_max_value) > span_prob_max_value):
					span_prob_max_value = end_probs[i] + start_prob_max_value
					span_prob_max_indices[0] = start_prob_max_index
					span_prob_max_indices[1] = i

			best_span[batch][0] = span_prob_max_indices[0]
			best_span[batch][1] = span_prob_max_indices[1]
		return best_span #best_span: batch_size*2

if __name__=='__main__':

	dict_args = {
				 'use_charemb': True,
				 'charvocab_size':20,
				 'charemb_dim':5,
				 'charemb_rnn_hdim':5,
				 'charemb_rnn_type':'LSTM',
				 'charemb_padix':0,
				 'wordemb_dim':10,
				 'contextemb_rnn_hdim':10,
				 'contextemb_rnn_type':'LSTM',
				 'contextemb_num_layers':3,
				 'bidaf_similarity_function': 'WeightedInputsDotConcatenation',
				 'modelinglayer_rnn_type':'LSTM',
				 'modelinglayer_one_num_layers':2,
				 'modelinglayer_two_num_layers':1,
				 'dropout_rate':0.2
				}
	bidaf = BiDAF(dict_args)
	passage = Variable(torch.randn(2,3,10))
	passage_lengths = torch.LongTensor([3,2])
	passagechars = Variable(torch.LongTensor([[[1,3,0,0],[5,6,7,0],[4,3,12,13]],[[11,2,18,8],[12,14,15,10],[0,0,0,0]]]))
	passagechar_lengths = torch.LongTensor([[2,3,4],[4,4,1]])
	question = Variable(torch.randn(2,2,10))
	question_lengths = torch.LongTensor([1,2])
	questionchars = Variable(torch.LongTensor([[[7,4,0,0],[0,0,0,0]],[[18,12,8,0],[1,1,2,0]]]))
	questionchar_lengths = torch.LongTensor([[2,1],[3,3]])
	start_indices = Variable(torch.LongTensor([2,0]))
	end_indices = Variable(torch.LongTensor([2,1]))

	start_index_probabilities, end_index_probabilities = \
		bidaf(passage, passage_lengths, passagechars, passagechar_lengths,\
			  question, question_lengths, questionchars, questionchar_lengths)
	print(start_index_probabilities)
	print(end_index_probabilities)
	#Why are the probabilities evenly distributed?

	print(bidaf.find_span(start_index_probabilities.data.cpu(), end_index_probabilities.data.cpu()))

