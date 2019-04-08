import sys
import re
import pickle
import pprint

en_words = {}
fr_words = {}

i_en = 0 
i_fr = 0 

# I don't want to create key for those containing punctuations, but this will cause problem for the next step
# punct_list = [",", "'", "\"", "~", "`", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "=", "+", "[", "{", "]", "}", "|", ";", ":", "<", ".", ">", "/", "?"]

# all words are lowercased

# ---------------------------------------------------- build dict[word]=word_id

with open("../txt_res/1.lexweights") as direct:  # translate En word to Fr words
	for line in direct:
		line = line.strip('\n')
		# can't only use word 'entropy' to identify boundaries between word entries
		if "nTrans" in line:
			en_token = line.split('\t')[0]
			if en_token not in en_words:
				en_words[en_token] = i_en
				i_en += 1 
		else:
			# ':' is the delimiter between translation and the proba
			if re.search(' ::', line):
				fr_token = ':'
			else:
				fr_token = line.split(': ')[0].lstrip()

			if fr_token not in fr_words:
				fr_words[fr_token] = i_fr
				i_fr += 1

with open("../txt_res/2.lexweights") as reverse:   # continue to construct word->id dictionaries
	for line in reverse:
		line = line.strip('\n')
		if "nTrans" in line:
			fr_token = line.split('\t')[0]
			if fr_token not in fr_words:
				fr_words[fr_token] = i_fr
				i_fr += 1
		else:
			if re.search(' ::', line):
				en_token = ':'
			else:
				en_token = line.split(': ')[0].lstrip()

			if en_token not in en_words:
				en_words[en_token] = i_en
				i_en += 1

# print(len(en_words))
# print(len(fr_words))
#
# # ------------------------------------------------------------------build entropy and lexical translation table
#
dic_direct = {}   # dic_direct[en_wordID] = {fr_wordID:proba, ...}
dic_reverse = {}

en_entropy = {}  #  translation entropy for an English word
fr_entropy = {}

with open("../txt_res/1.lexweights") as direct:
	for line in direct:
		line = line.strip('\n')
		if "nTrans" in line:
			en_token = line.split('\t')[0]
			entropy = line.split('\t')[1].split(' ')[1]
			en_id = en_words[en_token]

			en_entropy[en_id] = entropy
			translations = {}
			dic_direct[en_id] = translations
		else:
			if re.search(' ::', line):
				fr_id = fr_words[':']
				proba = line.lstrip(' :: ')
				translations[fr_id] = proba
			else:
				if ':' in line:
					# split by the delimiter ': ' instead of only ':'
					# to handle correctly this case -> 2:00: 0,000003
					fr_token = line.split(': ')[0].lstrip()
					if fr_token != '':
						fr_id = fr_words[fr_token]
						proba = line.split(': ')[1].lstrip()
						translations[fr_id] = proba
					else:
						sys.stdout.write('')
				else:
					sys.stdout.write('')

with open("../txt_res/2.lexweights") as reverse:
	for line in reverse:
		line = line.strip('\n')
		if "nTrans" in line:
			fr_token = line.split('\t')[0]
			entropy = line.split('\t')[1].split(' ')[1]
			fr_id = fr_words[fr_token]

			fr_entropy[fr_id] = entropy
			translations = {}
			dic_reverse[fr_id] = translations
		else:
			if re.search(' ::', line):
				en_id = en_words[':']
				proba = line.lstrip(' :: ')
				translations[en_id] = proba
			else:
				if ':' in line:
					en_token = line.split(': ')[0].lstrip()
					if en_token != '':
						en_id = en_words[en_token]
						proba = line.split(': ')[1].lstrip()
						translations[en_id] = proba
					else:
						sys.stdout.write('')
				else:
					sys.stdout.write('')


pickle.dump(en_words, open("../pickle_res/en_word_id.p", "wb"))
pickle.dump(fr_words, open("../pickle_res/fr_word_id.p", "wb"))

pickle.dump(en_entropy, open("../pickle_res/en_entropy.p", "wb"))
pickle.dump(fr_entropy, open("../pickle_res/fr_entropy.p", "wb"))

pickle.dump(dic_direct, open("../pickle_res/berkeley_forward_table.p", "wb"))
pickle.dump(dic_reverse, open("../pickle_res/berkeley_reverse_table.p", "wb"))


