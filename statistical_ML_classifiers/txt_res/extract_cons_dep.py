import re
import pickle
import os

# os.mkdir("./en_cons_parsing/")
# os.mkdir("./en_dep_parsing/")
os.mkdir("./fr_cons_parsing/")
# os.mkdir("./fr_dep_parsing/")

#---------- separate English cons & dep parsing information for each sentence

en_parsing_dir = "./en_parsing/"

for i in range(1, len(os.listdir(en_parsing_dir))+1):
    enParsing = open(en_parsing_dir + str(i) + ".txt", "r")
    enString = enParsing.read()

    enCons = enString.split("Constituency parse: ")[1].split("Dependency Parse (enhanced plus plus dependencies):")[0]
    enDep = enString.split("Constituency parse: ")[1].split("Dependency Parse (enhanced plus plus dependencies):")[1]

    consFile = open("./en_cons_parsing/" + str(i) + ".txt", "w")
    print(enCons, file=consFile)

    depFile = open("./en_dep_parsing/" + str(i) + ".txt", "w")
    print(enDep, file=depFile)

    enParsing.close()
    consFile.close()
    depFile.close()

#---------- separate French dep parsing information for each sentence

fr_parsing_dir = "./fr_parsing/"

for i in range(1, len(os.listdir(fr_parsing_dir))+1):
    frParsing = open(fr_parsing_dir + str(i) + ".txt", "r")
    frString = frParsing.read()

    frDep = frString.split("Constituency parse: ")[1].split("Dependency Parse (enhanced plus plus dependencies):")[1]
    depFile = open("./fr_dep_parsing/" + str(i) + ".txt", "w")
    print(frDep, file=depFile)

    frParsing.close()
    depFile.close()

#---------- separate French const parsing information for each sentence

fr_cons_file = "./fr_noID_const_bonsai.txt"
i = 1
with open(fr_cons_file, "r") as input:
    for line in input:
        consFile = open("./fr_cons_parsing/" + str(i) + ".txt", "w")
        line = line.split('\n')
        line = line[0] # => very strange, each line is a table of two items, the first is the line content, the second is an empty string
        print(line, file=consFile)
        i += 1
        consFile.close()