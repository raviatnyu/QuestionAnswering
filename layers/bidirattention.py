import torch
import torch.nn as nn
from torch.autograd import Variable

import layers.utils as utils
import layers.similarity as similarity

class BiDirAttention(nn.Module):

	def __init__(self, dict_args):
		super(BiDirAttention, self).__init__()
		self.similarity_function_name = dict_args['similarity_function']
		#attends a sequence(instead of word) over an other sequence 
		self.one_shot_attention = dict_args['one_shot_attention']
		self.self_match_attention = dict_args['self_match_attention']
		self.similarity_function = None
		if self.similarity_function_name == 'DotProduct':
			self.similarity_function = similarity.DotProductSimilarity(dict_args)
		elif self.similarity_function_name == 'WeightedInputsConcatenation':
			self.similarity_function = similarity.LinearConcatenationSimilarity(dict_args)
		elif self.similarity_function_name == 'WeightedInputsDotConcatenation':
			self.similarity_function = similarity.LinearConcatenationDotSimilarity(dict_args)
		elif self.similarity_function_name == 'WeightedSumProjection':
			self.similarity_function = similarity.LinearProjectionSimilarity(dict_args)
		elif self.similarity_function_name == 'ProjectionSimilaritySharedWeights':
			self.similarity_function = dict_args['similarity_function_pointer']


	def forward(self, sequence1, sequence2, sequence1_mask, sequence2_mask):
		#sequence1: batch_size*num_words1*iembed1
		#sequence2: batch_size*num_words2*iembed2
		if self.similarity_function_name == 'DotProduct':
			assert (sequence1.size(2) == sequence2.size(2)),"iembed1 and iembed2 should be same for dotproduct similarity" 

		sequence1_tiled = sequence1.unsqueeze(2).expand(sequence1.size(0), sequence1.size(1), sequence2.size(1), sequence1.size(2))
		#sequence1_tiled: batch_size*num_words1*num_words2*iembed1
		sequence2_tiled = sequence2.unsqueeze(1).expand(sequence2.size(0), sequence1.size(1), sequence2.size(1), sequence2.size(2))
		#sequence2_tiled: batch_size*num_words1*num_words2*iembed2
		similarity_matrix = self.similarity_function(sequence1_tiled, sequence2_tiled) #similarity_matrix: batch_size*num_words1*num_words2

		#sequence1 attending over sequence2
		similarity_matrix_shape = similarity_matrix.size()
		sequence2_attention_weights = utils.masked_softmax(similarity_matrix.view(-1, similarity_matrix_shape[-1]),\
			sequence2_mask.unsqueeze(1).expand_as(similarity_matrix).contiguous().view(-1, sequence2_mask.size(-1)).float())
		sequence2_attention_weights = sequence2_attention_weights.view(*similarity_matrix_shape) #sequence2_attention_weights: batch_size*num_words1*num_words2
		#sequence2 attending over sequence1
		if not self.self_match_attention:
			if self.one_shot_attention == True:
				sequence1_attention_weights = (similarity_matrix*sequence2_mask.unsqueeze(1).float() + (1-sequence2_mask.unsqueeze(1).float())*-1e9).max(dim=-1)[0]
				#sequence1_attention_weights: batch_size*num_words1
				sequence1_attention_weights = utils.masked_softmax(sequence1_attention_weights, sequence1_mask.float()) #sequence1_attention_weights: batch_size*num_words1
			else:
				pass #similar to seq1 over seq2 copy when needed

		#sequence2 attention pooling for sequence1
		sequence1_sequence2_attention = utils.attention_pooling(sequence2_attention_weights, sequence2) #sequence1_sequence2_attention: batch_size*num_words1*iembed2
		#sequence1 attention pooling for sequence2
		if not self.self_match_attention:
			sequence2_sequence1_attention = utils.attention_pooling(sequence1_attention_weights, sequence1) #sequence2_sequence1_attention: batch_size*iembed1


		if not self.self_match_attention:
			return sequence1_sequence2_attention, sequence2_attention_weights, sequence2_sequence1_attention, sequence1_attention_weights #sequence2_sequence1_attention: batch_size*iembed1
		else:
			return sequence1_sequence2_attention, sequence2_attention_weights

if __name__=='__main__':
	#bidir = BiDirAttention({'similarity_function': 'DotProduct', 'one_shot_attention':True})
	bidir = BiDirAttention({'similarity_function': 'WeightedSumProjection', 'sequence1_dim':10, 'sequence2_dim':10, 'projection_dim':10, 'one_shot_attention':True, 'self_match_attention':False})
	sequence1_sequence2_attention, sequence2_attention_weights, sequence2_sequence1_attention, sequence1_attention_weights =\
		bidir(Variable(torch.randn(2,5,10)), Variable(torch.randn(2,3,10)),\
		Variable(utils.sequence_mask(torch.LongTensor([5,3]))), Variable(utils.sequence_mask(torch.LongTensor([3,2]))))
	print(sequence1_sequence2_attention)
	print(sequence2_attention_weights)
	print(sequence2_sequence1_attention)
	print(sequence1_attention_weights)

	'''bidir = BiDirAttention({'similarity_function': 'WeightedSumProjection', 'sequence1_dim':7, 'sequence2_dim':7, 'projection_dim':7, 'one_shot_attention':False, 'self_match_attention':True})
	sequence = Variable(torch.randn(2,5,7))
	sequence_sequence_attention, sequence_attention_weights = bidir(sequence, sequence,\
		Variable(utils.sequence_mask(torch.LongTensor([5,3]))), Variable(utils.sequence_mask(torch.LongTensor([5,3]))))
	print(sequence_sequence_attention)
	print(sequence_attention_weights)'''

	'''bidir = BiDirAttention({'similarity_function': 'WeightedInputsConcatenation', 'input_dim': 10})
	print(bidir(Variable(torch.randn(2,5,10)), Variable(torch.randn(2,3,10))))
	bidir = BiDirAttention({'similarity_function': 'WeightedInputsDotConcatenation', 'input_dim': 10})
	print(bidir(Variable(torch.randn(2,5,10)), Variable(torch.randn(2,3,10))))'''


