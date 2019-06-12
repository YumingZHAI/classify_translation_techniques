import re, os

# transform EN, FR constituent parsing format to one word per line, and add word indice
os.mkdir("en_cons_transform/")
os.mkdir("fr_cons_transform/")

terminal_node_regex = r'\([A-Z$\+]+ [\w\°\%\`\'\,\.\?\!\"\:\;\-(\-LRB\-)(\-RRB\-)]+\)'

# we need to map these two tag sets to a smaller one
# before mapping:
# English tag set, apart from [A-Z]+ ones: PRP$, WP$; use punctuations instead of the tag PONCT ;
# French tag: P+D French tag

for i in range(1, len(os.listdir("en_cons_parsing"))+1):
	output = open("en_cons_transform/" + str(i) + ".txt", "w")
	with open("en_cons_parsing/" + str(i) + ".txt", "r") as file:
		string = file.read()
		# they use ex. (, ,) instead of PONCT (bonsai uses PONCT), which makes the token indice count false
		string = re.sub(r'\([\`\'\,\.\?\!\"\:\;\-]+', '(PONCT', string)
		# search: (V peut), (P pendant), (VPP intéressé), (PONCT ,)
		tab = re.findall(terminal_node_regex, string, re.UNICODE)
		# print(tab)
		for item in tab:
			# make one word per line
			string = string.replace(item, '\n'+item+'\n')
		newtab = string.split('\n')
		outlist = []
		for item in newtab:
			# remove empty lines
			if '(' in item:
				outlist.append(item)
		# add word index
		i = 1  # for token
		j = 0  # for line
		for line in outlist:
			if re.match(terminal_node_regex, line, re.UNICODE):
				# change directly in the list
				outlist[j] = line + "_" + str(i)
				i += 1
			j += 1

		for line in outlist:
			print(line, file=output)
		output.close()

#------------------------------------------------------------------------

for i in range(1, len(os.listdir("fr_cons_parsing"))+1):
	output = open("fr_cons_transform/" + str(i) + ".txt", "w")
	with open("fr_cons_parsing/" + str(i) + ".txt", "r") as file:
		string = file.read()
		# search: (V peut), (P pendant), (VPP intéressé), (PONCT ,)
		tab = re.findall(terminal_node_regex, string, re.UNICODE)
		for item in tab:
			# make one word per line
			string = string.replace(item, '\n'+item+'\n')
		newtab = string.split('\n')
		outlist = []
		for item in newtab:
			# remove empty lines
			if '(' in item:
				outlist.append(item)
		# add word index
		i = 1  # for token
		j = 0  # for line
		for line in outlist:
			if re.match(terminal_node_regex, line, re.UNICODE):
				# change directly in the list
				outlist[j] = line + "_" + str(i)
				i += 1
			j += 1

		for line in outlist:
			print(line, file=output)
		output.close()

