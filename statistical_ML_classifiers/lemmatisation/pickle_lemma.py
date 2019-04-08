import pickle
import sys
import pprint

eng_lemma = {}
fr_lemma = {}

# get the list of lineIDs
list = []
with open("../txt_res/en.txt") as all:
	for line in all:
		line = line.strip('\n')
		lineID = line.split(' ')[0]
		list.append(lineID)

i = 0
with open("ENlemme_from_corenlp.txt") as englemma, open("fr_lemma.txt") as frlemma:
	for l1, l2 in zip(englemma, frlemma):
		l1 = l1.strip('\n')
		l2 = l2.strip('\n')
		tab1 = l1.split(' ')		
		tab2 = l2.split(' ')
		# eng/fr_lemma[line ID] = table of lemma
		if (i < len(list)):
			eng_lemma[int(list[i])] = tab1
			fr_lemma[int(list[i])] = tab2
		i += 1 

# pprint.pprint(eng_lemma, width=1)
# print(eng_lemma[515])

pickle.dump(eng_lemma, open("../pickle_res/eng_lemma.p", "wb"))
pickle.dump(fr_lemma, open("../pickle_res/fr_lemma.p", "wb"))


