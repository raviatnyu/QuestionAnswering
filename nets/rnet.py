import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional

from layers import RNNCharEmb
from layers import BiDirEncoder
from layers import BiDirAttention
from layers import SelfAttention
from layers import UniDirAttention
from layers import MatchAttention
from layers import PointerNetwork
from layers import Gate
from layers import utils as utils
from layers import similarity as similarity

class RNet(nn.Module):

	def __init__(self, dict_args):
		super(RNet, self).__init__()

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

		#gated attention layer
		self.gated_attention_similarity_function = dict_args['gated_attention_similarity_function']
		self.gated_attentio_rnn_type = dict_args['gated_attentio_rnn_type']

		#self matching layer
		self.self_matching_similarity_function = dict_args['self_matching_similarity_function']

		#modeling layer
		self.modelinglayer_rnn_hdim = dict_args['contextemb_rnn_hdim']
		self.modelinglayer_rnn_type = dict_args['modelinglayer_rnn_type']
		self.modelinglayer_num_layers = dict_args['modelinglayer_num_layers']

		#question attention layer
		self.question_attention_similarity_function = dict_args['question_attention_similarity_function']

		#pointer network layer
		self.pointer_network_similarity_function = dict_args['pointer_network_similarity_function']
		self.pointer_network_rnn_type = dict_args['pointer_network_rnn_type']

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

		#######context embedding layer
		contextemb_layer_args = {
								   'input_dim': 2*self.charemb_rnn_hdim + self.wordemb_dim , 
								   'rnn_hdim': self.contextemb_rnn_hdim,  
								   'rnn_type': self.contextemb_rnn_type, 
								   'num_layers': self.contextemb_num_layers,
								   'dropout_rate' : self.dropout_rate 
								}
		self.contextemb_layer = BiDirEncoder(contextemb_layer_args)

		#######gated attention layer
		self.projection_dim =  2*self.contextemb_rnn_hdim
		self.question_projection_weights_u = nn.Parameter(torch.Tensor(self.projection_dim, 2*self.contextemb_rnn_hdim))
		self.passage_projection_weights_u = nn.Parameter(torch.Tensor(self.projection_dim, 2*self.contextemb_rnn_hdim))
		self.passage_projection_weights_v = nn.Parameter(torch.Tensor(self.projection_dim, 2*self.contextemb_rnn_hdim))
		self.dotproduct_weights = nn.Parameter(torch.Tensor(self.projection_dim,1))

		stdv = 1.0 / math.sqrt(self.question_projection_weights_u.size(-1))
		self.question_projection_weights_u.data.uniform_(-stdv, stdv)
		stdv = 1.0 / math.sqrt(self.passage_projection_weights_u.size(-1))
		self.passage_projection_weights_u.data.uniform_(-stdv, stdv)
		stdv = 1.0 / math.sqrt(self.passage_projection_weights_v.size(-1))
		self.passage_projection_weights_v.data.uniform_(-stdv, stdv)
		stdv = 1.0 / math.sqrt(self.dotproduct_weights.size(-1))
		self.dotproduct_weights.data.uniform_(-stdv, stdv)

		gated_similarity_function_args = {
											'projection_dim' : self.projection_dim,
											'sequence1_weights' : self.question_projection_weights_u,
											'sequence2_weights' : self.passage_projection_weights_u,
											'sequence3_weights' : self.passage_projection_weights_v,
											'weights' : self.dotproduct_weights
										 }
		self.gated_similarity_function_pointer = similarity.ProjectionSimilaritySharedWeights(gated_similarity_function_args)

		gated_attention_layer_args = {
										'similarity_function': self.gated_attention_similarity_function,
										'similarity_function_pointer': self.gated_similarity_function_pointer, 
										'sequence1_dim': 2*self.contextemb_rnn_hdim, 
										'sequence2_dim': 2*self.contextemb_rnn_hdim,
										'rnn_type': self.gated_attentio_rnn_type, 
										'rnn_hdim': 2*self.contextemb_rnn_hdim, 
										'gated_attention':dict_args['use_gating']
									 }
		self.use_bidirectional = dict_args['use_bidirectional']		
		self.gated_attention_layer_forward = MatchAttention(gated_attention_layer_args)
		self.gatedattn_dim = 2*self.contextemb_rnn_hdim
		if self.use_bidirectional:
			self.gated_attention_layer_backward = MatchAttention(gated_attention_layer_args)
			self.gatedattn_dim = 4*self.contextemb_rnn_hdim

		#######self matching layer
		self.use_selfmatching = dict_args['use_selfmatching']
		self.selfmatching_dim = 0
		if self.use_selfmatching:
			#self.projection_dim =  2*self.contextemb_rnn_hdim #Shared
			self.passage_projection_weights_v1 = nn.Parameter(torch.Tensor(self.projection_dim, self.gatedattn_dim)) #Shared
			self.passageprime_projection_weights_v = nn.Parameter(torch.Tensor(self.projection_dim, self.gatedattn_dim))
			self.dotproduct_weights1 = nn.Parameter(torch.Tensor(self.projection_dim,1)) #Shared

			stdv = 1.0 / math.sqrt(self.passageprime_projection_weights_v.size(-1))
			self.passageprime_projection_weights_v.data.uniform_(-stdv, stdv)
			stdv = 1.0 / math.sqrt(self.passage_projection_weights_v1.size(-1))
			self.passage_projection_weights_v1.data.uniform_(-stdv, stdv)
			stdv = 1.0 / math.sqrt(self.dotproduct_weights1.size(-1))
			self.dotproduct_weights1.data.uniform_(-stdv, stdv)

			selfmatching_similarity_function_args = {
														'projection_dim' : self.projection_dim,
														'sequence1_weights' : self.passage_projection_weights_v1,
														'sequence2_weights' : self.passageprime_projection_weights_v,
														'weights' : self.dotproduct_weights1
											 		}
			self.selfmatching_similarity_function_pointer = similarity.ProjectionSimilaritySharedWeights(selfmatching_similarity_function_args)	

			self_matching_layer_args = {
											'similarity_function': self.self_matching_similarity_function,
											'similarity_function_pointer': self.selfmatching_similarity_function_pointer,
											'sequence_dim': 2*self.contextemb_rnn_hdim,	
											'projection_dim': 2*self.contextemb_rnn_hdim
							   		   }
			self.self_matching_layer = SelfAttention(self_matching_layer_args)
			self.selfmatching_dim = self.gatedattn_dim

		#######Gated layer
		self.gated_selfmatching = self.use_selfmatching and dict_args['use_gating']
		if self.gated_selfmatching:
			self.gate_dim = self.gatedattn_dim + self.selfmatching_dim
			self.selfmatchinggate = Gate({'sigmoidinputdim':self.gate_dim, 'gateinputdim':self.gate_dim})		



		#######modeling layer
		modeling_layer_args = {
								'input_dim': self.gatedattn_dim + self.selfmatching_dim, 
								'rnn_hdim': self.modelinglayer_rnn_hdim,  
								'rnn_type': self.modelinglayer_rnn_type, 
								'num_layers': self.modelinglayer_num_layers,
								'dropout_rate' : 0 
							  }
		self.modeling_layer = BiDirEncoder(modeling_layer_args)

		#######question attention layer
		self.question_query_vector = nn.Parameter(torch.Tensor(2*self.contextemb_rnn_hdim))
		stdv = 1.0 / math.sqrt(self.question_query_vector.size(-1))
		self.question_query_vector.data.uniform_(-stdv, stdv)

		#self.projection_dim =  2*self.contextemb_rnn_hdim #Shared
		self.question_projection_weights_u2 = nn.Parameter(torch.Tensor(self.projection_dim, 2*self.contextemb_rnn_hdim)) #Shared
		self.question_projection_weights_v = nn.Parameter(torch.Tensor(self.projection_dim, 2*self.contextemb_rnn_hdim))
		self.dotproduct_weights2 = nn.Parameter(torch.Tensor(self.projection_dim,1)) #Shared

		stdv = 1.0 / math.sqrt(self.question_projection_weights_v.size(-1))
		self.question_projection_weights_v.data.uniform_(-stdv, stdv)
		stdv = 1.0 / math.sqrt(self.question_projection_weights_u2.size(-1))
		self.question_projection_weights_u2.data.uniform_(-stdv, stdv)
		stdv = 1.0 / math.sqrt(self.dotproduct_weights2.size(-1))
		self.dotproduct_weights2.data.uniform_(-stdv, stdv)

		question_similarity_function_args = {
													'projection_dim' : self.projection_dim,
													'sequence1_weights' : self.question_projection_weights_u2,
													'sequence2_weights' : self.question_projection_weights_v,
													'weights' : self.dotproduct_weights2
										 		}
		self.question_similarity_function_pointer = similarity.ProjectionSimilaritySharedWeights(question_similarity_function_args)

		question_attention_layer_args = {
										  'similarity_function': self.question_attention_similarity_function,
										  'similarity_function_pointer': self.question_similarity_function_pointer,
										  'sequence1_dim': 2*self.contextemb_rnn_hdim,
										  'sequence2_dim': 2*self.contextemb_rnn_hdim,
										  'projection_dim': 2*self.contextemb_rnn_hdim
										}
		self.question_attention_layer = UniDirAttention(question_attention_layer_args)

		#######pointer network layer
		#self.projection_dim =  2*self.contextemb_rnn_hdim #Shared
		self.passage_projection_weights_h = nn.Parameter(torch.Tensor(self.projection_dim, 2*self.contextemb_rnn_hdim))
		self.decoder_projection_weights_h = nn.Parameter(torch.Tensor(self.projection_dim, 2*self.contextemb_rnn_hdim))
		self.dotproduct_weights3 = nn.Parameter(torch.Tensor(self.projection_dim,1)) #Shared

		stdv = 1.0 / math.sqrt(self.passage_projection_weights_h.size(-1))
		self.passage_projection_weights_h.data.uniform_(-stdv, stdv)
		stdv = 1.0 / math.sqrt(self.decoder_projection_weights_h.size(-1))
		self.decoder_projection_weights_h.data.uniform_(-stdv, stdv)
		stdv = 1.0 / math.sqrt(self.dotproduct_weights3.size(-1))
		self.dotproduct_weights3.data.uniform_(-stdv, stdv)

		pointer_similarity_function_args = {
													'projection_dim' : self.projection_dim,
													'sequence1_weights' : self.passage_projection_weights_h,
													'sequence2_weights' : self.decoder_projection_weights_h,
													'weights' : self.dotproduct_weights3
										 	}
		self.pointer_similarity_function_pointer = similarity.ProjectionSimilaritySharedWeights(pointer_similarity_function_args)

		pointer_network_layer_args = {
										'similarity_function': self.pointer_network_similarity_function,
										'similarity_function_pointer': self.pointer_similarity_function_pointer,
										'sequence_dim': 2*self.contextemb_rnn_hdim,
										'projection_dim': 2*self.contextemb_rnn_hdim,
										'rnn_type': self.pointer_network_rnn_type,
										'rnn_hdim': 2*self.contextemb_rnn_hdim,
									 }
		self.pointer_network_layer = PointerNetwork(pointer_network_layer_args)


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

		passage_question_matchattn_forward, question_matchattn_weights_forward = \
			self.gated_attention_layer_forward(passage_embedding, question_embedding, passage_mask, question_mask)
		#passage_question_matchattn_forward: batch_size*num_words_passage*2.contextemb_rnn_hdim
		#question_matchattn_weights_forward: batch_size*num_words_passage*num_words_question
		passage_question_matchattn = passage_question_matchattn_forward
		if self.use_bidirectional:
			passage_question_matchattn_reverse, question_matchattn_weights_reverse = \
				self.gated_attention_layer_backward(utils.reverse_sequence(passage_embedding), question_embedding,\
													utils.reverse_sequence(passage_mask), question_mask, reverse=True)
			passage_question_matchattn_reverse = utils.reverse_sequence(passage_question_matchattn_reverse)
			if not self.training: question_matchattn_weights_reverse = utils.reverse_sequence(question_matchattn_weights_reverse)
			#passage_question_matchattn_reverse: batch_size*num_words_passage*2.contextemb_rnn_hdim
			passage_question_matchattn = torch.cat((passage_question_matchattn_forward, passage_question_matchattn_reverse), dim=-1)
			#passage_question_matchattn: batch_size*num_words_passage*4.contextemb_rnn_hdim

		question_aware_passage_representation = passage_question_matchattn

		if self.use_selfmatching:
			passage_passage_selfattn, passage_selfattn_weights = \
				self.self_matching_layer(passage_question_matchattn, passage_mask)
			#passage_passage_selfattn: batch_size*num_words_passage*2.contextemb_rnn_hdim
			#passage_selfattn_weights: batch_size*num_words_passage*num_words_passage
			#Skipping masking for passage_passage_selfattn
			question_aware_passage_representation = torch.cat((passage_question_matchattn, passage_passage_selfattn), dim = -1)

		#if self.use_dropout: question_aware_passage_representation = self.dropout_layer(question_aware_passage_representation)
		if self.gated_selfmatching:
			question_aware_passage_representation = self.selfmatchinggate(question_aware_passage_representation, question_aware_passage_representation)
		
		question_aware_passage_representation, _ = self.modeling_layer(question_aware_passage_representation, passage_lengths)
		#question_aware_passage_representation: batch_size*num_words_passage*2.self.modelinglayer_rnn_hdim

		if self.use_dropout: question_aware_passage_representation = self.dropout_layer(question_aware_passage_representation)

		self.question_query_vector_expand = self.question_query_vector.unsqueeze(0).expand(question_embedding.size(0), self.question_query_vector.size(0))
		#self.question_query_vector_expand: batch_size*2.self.contextemb_rnn_hdim	(question_query_vector_dim = 2.self.contextemb_rnn_hdim)
		queryvector_question_attention, question_attention_weights = \
			self.question_attention_layer(question_embedding, self.question_query_vector_expand, question_mask)
		#queryvector_question_attention: batch_size*2.contextemb_rnn_hdim
		#question_attention_weights: batch_size*num_words_question

		pointer_passage_attention_values = \
			self.pointer_network_layer(question_aware_passage_representation, queryvector_question_attention, passage_mask, 2)
		#pointer_passage_attention_probabilities: batch_size*2*num_words_passage

		start_index_values, end_index_values = torch.chunk(pointer_passage_attention_values, 2, 1)
		start_index_values = start_index_values.squeeze()
		end_index_values = end_index_values.squeeze()

		'''start_index_probabilities = utils.masked_softmax(start_index_values, passage_mask.float())
		end_index_probabilities = utils.masked_softmax(end_index_values, passage_mask.float())'''

		start_index_log_probabilities = start_index_values + passage_mask.float().log()
		end_index_log_probabilities = end_index_values + passage_mask.float().log()
		start_index_log_probabilities = functional.log_softmax(start_index_log_probabilities)
		end_index_log_probabilities = functional.log_softmax(end_index_log_probabilities)
		#start_index_log_probabilities: batch_size*num_words_passage
		#end_index_log_probabilities: batch_size*num_words_passage

		#uncomment the below line in compare mode
		#return start_index_log_probabilities, end_index_log_probabilities, (question_matchattn_weights_forward + question_matchattn_weights_reverse)/2
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
				 'gated_attention_similarity_function': 'ProjectionSimilaritySharedWeights',
				 'gated_attentio_rnn_type': 'LSTM',
				 'use_bidirectional': True,
				 'use_gating': True,
				 'use_selfmatching': True,
				 'self_matching_similarity_function': 'ProjectionSimilaritySharedWeights',
				 'modelinglayer_rnn_type':'LSTM',
				 'modelinglayer_num_layers':1,
				 'question_attention_similarity_function': 'ProjectionSimilaritySharedWeights',
				 'pointer_network_similarity_function': 'ProjectionSimilaritySharedWeights',
				 'pointer_network_rnn_type':'GRU',
				 'dropout_rate':0.2 
				}
	rnet = RNet(dict_args)
	passage = Variable(torch.randn(2,5,10))
	passage_lengths = torch.LongTensor([5,3])
	passagechars = Variable(torch.LongTensor([[[1,3,0,0],[5,6,7,0],[4,3,12,13],[11,2,7,8],[13,15,2,0]],[[11,2,18,8],[12,14,15,10],[1,6,3,0],[0,0,0,0],[0,0,0,0]]]))
	passagechar_lengths = torch.LongTensor([[2,3,4,4,3],[4,4,3,1,1]])
	question = Variable(torch.randn(2,2,10))
	question_lengths = torch.LongTensor([1,2])
	questionchars = Variable(torch.LongTensor([[[7,4,0,0],[0,0,0,0]],[[18,12,8,0],[1,1,2,0]]]))
	questionchar_lengths = torch.LongTensor([[2,1],[3,3]])
	start_indices = Variable(torch.LongTensor([2,0]))
	end_indices = Variable(torch.LongTensor([2,1]))

	start_index_log_probabilities, end_index_log_probabilities = \
		rnet(passage, passage_lengths, passagechars, passagechar_lengths,\
			  question, question_lengths, questionchars, questionchar_lengths)
	print(start_index_log_probabilities)
	print(end_index_log_probabilities)

	print(rnet.find_span(start_index_log_probabilities.data.cpu(), end_index_log_probabilities.data.cpu()))

