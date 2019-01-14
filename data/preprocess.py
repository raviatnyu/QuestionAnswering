from preprocessutils import *
from nltk import word_tokenize
from nltk import sent_tokenize

import os
import json

def process_tokens(tokenlist):
	tokens = []
	for token in tokenlist:
		if token is not "":
			token = token.replace("''","\"")
			token = token.replace("``","\"")
			tokens.append(token)
	return tokens

def process_text(text):
	text = text.replace("`","'")
	text = text.replace("''","\"")
	if "\n" in text:
		text = text.replace("\n","n")
	return text

def process_question(question):
	question = process_text(question)
	question_tokens = extra_tokenize(question)
	question_tokens = process_tokens(question_tokens)
	return {'question_tokens':question_tokens}

def process_answer(start_char, answertext, passagetext, passagetokens):
	passagetext = process_text(passagetext)
	end_char = start_char + len(answertext) - 1
	token_lengths = [len(token) for token in passagetokens]
	token_cum_lengths = [0 for lengthindex in range(len(token_lengths)+1)]
	token_cum_lengths[0] = 0
	for i in range(1, len(token_lengths)+1):
		token_cum_lengths[i] = token_cum_lengths[i-1] + token_lengths[i-1]
	num_start_spaces = passagetext[0:start_char+1].count(' ')
	num_end_spaces = num_start_spaces + answertext.count(' ')
	start_token = find_position(start_char-num_start_spaces, token_cum_lengths) -1
	end_token = find_position(end_char-num_end_spaces, token_cum_lengths) - 1
	return start_token, end_token

def process_answers(answerlist, passagetext, passagetokens):
	answers = []
	for answer in answerlist:
		text = answer['text']
		start_token, end_token = process_answer(answer['answer_start'], text, passagetext, passagetokens)
		answers.append({'span':(start_token,end_token), 'text':text})
	return answers

def process_qas(qalist, passagetext, passagetokens):
	qas = []
	for qa in qalist:
		question = process_question(qa['question'])
		instanceid = qa['id']
		answerlist =  process_answers(qa['answers'], passagetext, passagetokens)
		qas.append({'answers':answerlist,'question':question,'id':instanceid})
	return qas

def process_context(context):
	passage_tokens = []
	#passage_chars = []
	context = process_text(context)
	sentences = sent_tokenize(context)
	for sentence in sentences:
		#sentence_tokens = word_tokenize(sentence)
		sentence_tokens = extra_tokenize(sentence)
		passage_tokens = passage_tokens + sentence_tokens
	'''for token in passage_tokens:
		passage_chars.append(list(token))'''
	passage_tokens = process_tokens(passage_tokens)
	return {'passage_tokens':passage_tokens}

def process_paragraphs(paragraphlist):
	paragraphs = []
	for paragraph in paragraphlist:
		context = process_context(paragraph['context'])
		qalist = process_qas(paragraph['qas'], paragraph['context'], context['passage_tokens'])
		paragraphs.append({'context':context, 'qas':qalist})
	return paragraphs

def process_articles(articlelist):
	articles = []
	for article in articlelist:
		title = article['title']
		paragraphslist = process_paragraphs(article['paragraphs'])
		articles.append({'title':title,'paragraphs':paragraphslist})
	return articles

def process_data(filename):
	file = open(filename, 'r') 
	data = json.load(file)
	articleslist = process_articles(data['data'])
	input = {'data':articleslist}
	return input

if __name__=="__main__":
	traindata = json.dumps(process_data('train-v1.1.json'))
	devdata = json.dumps(process_data('dev-v1.1.json'))
	
	filespath = '../input'
	trainfilename = 'train.json'
	testfilename = 'dev.json'
	trainfilepath = os.path.join(filespath, trainfilename)
	devfilepath = os.path.join(filespath, testfilename)
	trainfile = open(trainfilepath,'w')
	devfile = open(devfilepath,'w')
	trainfile.write(traindata)
	devfile.write(devdata)

	'''devsplittokens = [
						'what', 'when', 'where', 'why', 'which', 'who', 'how', 'misc',\
						'len1', 'len2', 'len5', 'len10', 'len100'
					 ]
	for token in devsplittokens:
		devsplitfile = 'dev-v1.1' + '_' + token + '.json'
		devsplitdata = json.dumps(process_data(devsplitfile))
		devsplitfilename = 'dev' + '_' + token + '.json'
		devsplitfilepath = os.path.join(filespath, devsplitfilename)
		devsplitfileopen = open(devsplitfilepath,'w')
		devsplitfileopen.write(devsplitdata)
		devsplitfileopen.close()'''

