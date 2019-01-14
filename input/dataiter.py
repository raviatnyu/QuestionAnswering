import json

def squaditerdeep(data):
	for article in data['data']:
		for paragraph in article['paragraphs']:
			is_newparagraph = True
			passage_tokens = paragraph['context']['passage_tokens']
			for qa in paragraph['qas']:
				is_newquestion = True
				instance_id = qa['id']
				question_tokens = qa['question']['question_tokens']
				for answer in qa['answers']:
					answer_span = answer['span']
					if is_newparagraph:
						yield passage_tokens, question_tokens, answer_span, instance_id
					elif is_newquestion:
						yield None, question_tokens, answer_span, instance_id
					else:
						yield None, None, answer_span, None
					is_newparagraph = False
					is_newquestion = False

def squaditershallow(data):
	def get_answer_spans(answers_list):
		answer_spans = []
		for answer in answers_list:
			answer_spans.append(answer['span'])
		return answer_spans
	for article in data['data']:
		for paragraph in article['paragraphs']:
			is_newparagraph = True
			passage_tokens = paragraph['context']['passage_tokens']
			for qa in paragraph['qas']:
				instance_id = qa['id']
				answer_span = get_answer_spans(qa['answers'])
				question_tokens = qa['question']['question_tokens']
				if is_newparagraph:
					yield passage_tokens, question_tokens, answer_span, instance_id
				else:
					yield None, question_tokens, answer_span, instance_id
				is_newparagraph = False

def getinstanceinformation(data, instance_list):
	instance_info = []
	for article in data['data']:
		for paragraph in article['paragraphs']:
			passage_tokens = paragraph['context']['passage_tokens']
			for qa in paragraph['qas']:
				instance_id = qa['id']
				if instance_id in instance_list:
					question_tokens = qa['question']['question_tokens']
					for answer in qa['answers']:
						answer_span = answer['span']
						instance_info.append({'passage':passage_tokens, 'question':question_tokens, 'answer_span':answer_span, 'passage_len':len(passage_tokens) ,'id':instance_id})
	return instance_info
						 

if __name__=='__main__':
	instance_list = ['5725f4d2271a42140099d367', '57343f804776f41900661af9', '5726e259dd62a815002e93d6', '573034a1947a6a140053d28d', '5726a975708984140094cd38', '57265fb5dd62a815002e82fa', '56dfdbee7aa994140058e1ca', '5723f90ff6b826140030fd18', '573154bfa5e9cc1400cdbe96', '57267189708984140094c640', '570d2c0bb3d812140066d4e4', '56f9ecbff34c681400b0bee5', '57268a7ddd62a815002e88bf', '5727790a708984140094de94', '572ec29403f9891900756a02', '572658ebf1498d1400e8dcba', '56defc4bc65bf219000b3e87', '5707229c90286e26004fc947', '5727cc87ff5b5019007d957c', '57281ba43acd2414000df4b2', '56df60fc8bc80c19004e4b69', '572f6eacb2c2fd140056810a', '573605726c16ec1900b92904', '57324e68b9d445190005ea15', '572804ec3acd2414000df265', '572e8e2703f9891900756780', '5729a1066aef051400155050', '5728c8083acd2414000dfe3d', '570c55dab3d812140066d13c', '5726cd33f1498d1400e8eba0']
	data = json.load(open('train.json','r'))
	instance_info = getinstanceinformation(data, instance_list)
	for instance in instance_info:
		print(instance)
		print('##########################################################################################')

