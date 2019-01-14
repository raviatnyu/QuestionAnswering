# This Python file uses the following encoding: utf-8
from nltk import word_tokenize
from nltk import sent_tokenize

# Experiment
extra_token_splits = ['-', '/', '\u2013', '\u2014', '\u2212']

def extra_tokenize(sentence):
	tokens = []
	words = word_tokenize(sentence)
	for word in words:
		word = word.replace("``","\"")
		isSplitPresent = False
		for split in extra_token_splits:
			if split in word:
				subparts = word.split(split)
				isSplitPresent = True
				for index, subpart in enumerate(subparts):
					tokens.append(subpart)
					if index != len(subparts) - 1:
						tokens.append(split)
				break #ToDO
		if not isSplitPresent:
			tokens.append(word)
	return tokens

def find_position(value, cumlist):
	index = -1
	for i in range(len(cumlist)):
		if value < cumlist[i]:
			index = i
			break
	return index

# Put them in a list
def post_process(text):
	text = text.replace(' – ', '–')
	text = text.replace(' / ', '/')
	text = text.replace(' ,', ',')
	text = text.replace('\$ ', '\$')
	text = text.replace('( ', '(')
	text = text.replace(' )', ')')
	text = text.replace(" 's", "'s")
	text = text.replace(' °',  '°')
	text = text.replace(' %', '%')
	text = text.replace(' .', '.')
	text = text.replace(' !', '!')
	return text

if __name__ == '__main__':
	text = "100 – 5,000 hp God 's alone ! , ( God 's alone ) . 32 °C 67 % 54 % "
	print(text)
	print(post_process(text))

#Debugging to map char spans to word spans
	'''newanswertext = ''.join([passagetokens[index] for index in range(start_token,end_token+1)])
	var = (newanswertext.replace("'","").replace(".","").replace("~","").replace("£","") == answertext.replace(' ','').replace("'","").replace(".","").replace("~","").replace("£",""))
	if not var:
		print (str(var) + answertext + "-->" + newanswertext)
		print (passagetext)
		print (passagetokens)
		print (start_char-num_start_spaces)
		print (end_char-num_end_spaces)
		print (start_token)
		print (end_token)
		print (token_lengths)
		print (token_cum_lengths)
	else:
		print (str(var) + answertext + "-->" + newanswertext)'''
