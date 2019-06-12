import re, os

os.mkdir("en_parsing/")
os.mkdir("fr_parsing/")

## separate English const & dep parsing information from en_noID.txt.oneLine to en_parsing/files, one for each sentence
with open("en_noID.txt.oneLine", mode="r") as bigfile:
    allString = bigfile.read()
    # separator: ex. Sentence #2 (65 tokens):
    for i, part in enumerate(re.split(r'Sentence #\d.*', allString)):
        if i!=0:
            with open("en_parsing/" + str(i) + ".txt", mode="w") as newfile:
                print(part, file=newfile)

## separate French const & dep parsing information from fr_noID.txt.out to fr_parsing/files, one for each sentence
## and finally we only want dependency parsing information from corenlp output
with open("fr_noID.txt.out", mode="r") as bigfile:
    allString = bigfile.read()
    # separator: ex. Sentence #2 (65 tokens):
    for i, part in enumerate(re.split(r'Sentence #\d.*', allString)):
        if i!=0:
            with open("fr_parsing/" + str(i) + ".txt", mode="w") as newfile:
                print(part, file=newfile)