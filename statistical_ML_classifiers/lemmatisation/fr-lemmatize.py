import os, re

# at the end of sentence, don't put symbols like '#'，because tree-tagger(FR) will lemmatize it as <unknown>
os.system("sed -e 's/$/lemme/' ../txt_res/fr_noID.txt | tree-tagger-french > fr_tt.txt")

lemme_column = open("fr_lemma_column.txt", 'w')

with open("fr_tt.txt") as file:
    for line in file:
        line = line.strip('\n')
        tab = line.split('\t')
        word = tab[0]
        lemme = tab[2]
        # if lemmatised form is <unknown>, use its original form
        if lemme == '<unknown>':
            print(word, file=lemme_column)
        # if we can choose between two lemme candidats, choose the correct one
        elif re.search(r'\|', lemme):
            first = lemme.split('|')[0]
            second = lemme.split('|')[1]
            # these words are summarized from ted talks dev-test corpus
            if first in ['parent', 'échec', 'convenir', 'demi', 'fil', 'fonder', 'règle', 'lac', 'lire', 'plaire',
                         'poste', 'prochain', 'matériau', 'être']:
                print(first, file=lemme_column)
            else:
                print(second, file=lemme_column)
        elif lemme == '@card@':
            print(word, file=lemme_column)
        else:
            print(lemme, file=lemme_column)

lemme_column.close()

# attention for the '\\\n' to replace 'lemme'
os.system("cat fr_lemma_column.txt | tr '\n' ' ' | sed -e $'s/lemme/\\\n/g' | sed -e 's/^[ \t]*//' > fr_lemma.txt")
