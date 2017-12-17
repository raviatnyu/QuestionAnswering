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

def get_splitkey_passagelen(passage, splitkeys):
	passagelen = len(passage.split(' '))
	for splitkey in splitkeys:
		if passagelen <= int(splitkey):
			return splitkey
	return splitkeys[-1]

def process_paragraphs(paragraphlist, splitkeys, splitby, splitcounts):
	paragraphslistdict = initialize_dict(splitkeys)
	for paragraph in paragraphlist:
		context = paragraph['context']
		splitkey = get_splitkey_passagelen(context, splitkeys)
		paragraphslistdict[splitkey].append({'context':context, 'qas':paragraph['qas']})
		splitcounts[splitkey] += 1
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
	splitkeys = ['25', '50', '75', '100', '1000']
	splitby = 'passagelength'
	splitcounts = initialize_zerodict(splitkeys)
	splitjsonsdict = create_dev_split('dev-v1.1.json', splitkeys, splitby, splitcounts)
	print(splitcounts)
	for splitkey in splitkeys:
		filename = 'dev-v1.1' + '_len' + splitkey + '.json'
		devsplitfile = open(filename, 'w')
		devsplitfile.write(json.dumps(splitjsonsdict[splitkey]))
		devsplitfile.close()
