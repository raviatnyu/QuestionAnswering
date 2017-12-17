import os
import json
import statistics

def initialize_dict(splitkeys):
	listdict = {}
	for splitkey in splitkeys:
		listdict[splitkey] = []
	return listdict

def initialize_zerodict(splitkeys):
	listdict = {}
	for splitkey in splitkeys:
		listdict[splitkey] = 0
	return listdict

def get_splitkey_questiontype(question, splitkeys):
	for splitkey in splitkeys:
		if splitkey.lower() == question.split(' ')[0].lower():
			return splitkey
		'''if question.lower().startswith(splitkey.lower()):
			return splitkey'''
	return splitkeys[-1]

def get_splitkey_answerlen(answerlist, splitkeys):
	answer_lengths = []
	for answer in answerlist:
		answer_lengths.append(len(answer['text'].strip().split(' ')))
	for splitkey in splitkeys:
		if statistics.mean(answer_lengths) <= int(splitkey):
			return splitkey
	return splitkeys[-1]

def process_qas(qalist, splitkeys, splitby, splitcounts):
	qaslistdict = initialize_dict(splitkeys)
	for qa in qalist:
		question = qa['question']
		instanceid = qa['id']
		answerlist =  qa['answers']
		if splitby == 'questionstart':
			splitkey = get_splitkey_questiontype(question, splitkeys)
		elif splitby == 'answerlength':
			splitkey = get_splitkey_answerlen(answerlist, splitkeys)
		qaslistdict[splitkey].append({'answers':answerlist,'question':question,'id':instanceid})
		splitcounts[splitkey] += 1
	return qaslistdict

def process_paragraphs(paragraphlist, splitkeys, splitby, splitcounts):
	paragraphslistdict = initialize_dict(splitkeys)
	for paragraph in paragraphlist:
		context = paragraph['context']
		qalistdict = process_qas(paragraph['qas'], splitkeys, splitby, splitcounts)
		for splitkey in splitkeys:
			paragraphslistdict[splitkey].append({'context':context, 'qas':qalistdict[splitkey]})
	return paragraphslistdict

def process_articles(articlelist, splitkeys, splitby, splitcounts):
	articleslistdict = initialize_dict(splitkeys)
	for article in articlelist:
		title = article['title']
		paragraphslistdict = process_paragraphs(article['paragraphs'], splitkeys, splitby, splitcounts)
		for splitkey in splitkeys:
			articleslistdict[splitkey].append({'title':title,'paragraphs':paragraphslistdict[splitkey]})
	return articleslistdict

def create_dev_split(filename, splitkeys, splitby, splitcounts):
	input = {}
	file = open(filename, 'r') 
	data = json.load(file)
	articleslistdict = process_articles(data['data'], splitkeys, splitby, splitcounts)
	for splitkey in splitkeys:
		input[splitkey] = {'data':articleslistdict[splitkey]}
	return input

if __name__=='__main__':
	splitkeys = ['what', 'when', 'where', 'why', 'which', 'who', 'how', 'misc']
	splitby = 'questionstart'
	splitcounts = initialize_zerodict(splitkeys)
	splitjsonsdict = create_dev_split('dev-v1.1.json', splitkeys, splitby, splitcounts)
	print(splitcounts)
	for splitkey in splitkeys:
		filename = 'dev-v1.1' + '_' + splitkey + '.json'
		devsplitfile = open(filename, 'w')
		devsplitfile.write(json.dumps(splitjsonsdict[splitkey]))
		devsplitfile.close()
	splitkeys = ['1', '2', '5', '10', '100']
	splitby = 'answerlength'
	splitcounts = initialize_zerodict(splitkeys)
	splitjsonsdict = create_dev_split('dev-v1.1.json', splitkeys, splitby, splitcounts)
	print(splitcounts)
	for splitkey in splitkeys:
		filename = 'dev-v1.1' + '_len' + splitkey + '.json'
		devsplitfile = open(filename, 'w')
		devsplitfile.write(json.dumps(splitjsonsdict[splitkey]))
		devsplitfile.close()
