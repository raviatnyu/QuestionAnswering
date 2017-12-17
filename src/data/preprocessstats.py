from nltk import word_tokenize
from nltk import sent_tokenize

import unicodedata
import string
import json

all_letters = string.ascii_letters + " .,;:'"
n_letters = len(all_letters)

unicode_letters = ['S', 'u', 'p', 'e', 'r', ' ', 'B', 'o', 'w', 'l', '5', '0', 'a', 's', 'n', 'A', 'm', 'i', 'c', 'f', 't', 'b', 'g', 'd', 'h', 'N', 'F', 'L', '(', ')', '2', '1', '.', 'T', 'C', 'D', 'v', 'P', '4', '–', 'y', '7', ',', '6', "'", 'z', '"', '-', 'R', 'k', 'q', 'M', 'V', '9', 'G', 'E', 'X', 'I', '8', 'j', '½', 'U', '$', '3', 'é', ':', 'O', '/', 'H', 'x', 'J', 'W', 'K', ';', '#', 'Q', 'Y', '!', '&', '—', '[', ']', 'ł', 'Ż', 'ń', 'Ł', 'ó', '%', 'Z', '’', 'ę', 'ɑ', 'ː', 'ˈ', 'ʃ', 'ə', 'š', 'ś', 'ç', 'ü', 'ć', 'Ś', 'ʂ', 'ð', 'Î', 'ï', 'è', 'ú', 'á', 'Н', 'и', 'к', 'о', 'л', 'а', 'Т', 'е', 'с', 'Đ', 'č', '⅓', '?', 'ö', 'É', '{', '}', '×', '=', '+', '|', 'Π', '≠', '√', '⊆', '€', '£', 'θ', 'ɐ', '̯', 'ʊ', 'ä', '°', '′', '″', '‘', '−', '⁄', 'û', 'à', 'ß', '\n', 'σ', '*', 'π', 'ὀ', 'ξ', 'ύ', 'ς', 'γ', 'ε', 'ν', 'ή', 'ἄ', 'ζ', 'ω', 'τ', 'ο', '·', 'Å', '§', 'ô', 'ë', 'Ü', 'í', 'ê', 'õ', '±', 'ᵻ', 'ɒ', 'ɛ', 'ɔ', 'κ', 'ί', 'φ', 'έ', 'ρ', 'ō', '“', '”', '²', '<', '>', 'ø', 'ˌ', 'ɪ', 'æ', '…', 'ř', 'ā', 'ò', 'Ö', '₮', '正', '成', '吉', '思', '汗', 'ī', 'Ç', 'ı', 'ñ', 'ğ', 'Ч', 'н', 'г', 'х', 'Č', 'з', '铁', '木', '真', '鐵', '眞', 'ě', 'ù', 'ē', 'α', 'μ', 'ά', 'λ', 'η', 'ι', '`', '℞', 'β', 'δ', '\u200b', '元', '朝', '大', '《', '建', '國', '號', '詔', '》', '哉', '乾', '劉', '黑', '馬', '蕭', '札', '剌', '石', '抹', '孛', '迭', '兒', '塔', '不', '已', '刺', '之', '子', '重', '喜', '史', '秉', '直', '張', '柔', '嚴', '實', '都', '鈔', 'ạ', 'ằ', 'ầ', '陳', '京', 'ệ', 'ố', 'ấ', 'ư', '尚', '書', '省', '通', '制', '奎', '章', '閣', '學', '士', '院', '經', '世', '典', '太', '祖', '樞', '密', '授', '時', '暦', 'ɲ', 'ũ', 'å', '→', '≡', 'Ῥ', 'ῆ', '\ufeff', '~', 'إ', 'س', 'ل', 'ا', 'م', 'ي', '\u200e', 'œ']
all_letters = list(all_letters)
all_digits = ['0', '1', '2' , '3', '4', '5', '6', '7', '8', '9']
foreign_letters = list(set(unicode_letters)-set(all_letters)-set(all_digits))


def unicodeToAscii(s):
	return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def printUnicodeWords(filename):
	file = open(filename, 'r', encoding='utf-8') 
	data = json.load(file)
	characters = []
	for article in data['data']:
		for paragraph in article['paragraphs']:
			context = paragraph['context']
			words = context.split(" ")
			for word in words:
				for char in word:
					if char in foreign_letters:
						print (word)
						break
	return

def isWordNotUnicodeOnly(word):
	if (len(word) > 1):
		for char in word:
			if char in foreign_letters:
				return True
	return False

def printUntokenizedUnicodeWords(filename):
	file = open(filename, 'r', encoding='utf-8') 
	data = json.load(file)
	for article in data['data']:
		for paragraph in article['paragraphs']:
			context = paragraph['context']
			context = context.replace("`","'")
			context = context.replace("''","\"")
			sentences = sent_tokenize(context)
			for sentence in sentences:
				words = word_tokenize(sentence)
				for word in words:
					word = word.replace("``","\"")
					if isWordNotUnicodeOnly(word):
						print (word)
			break
	return
