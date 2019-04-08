import pickle

# lineID 1840 is in the first line, etc.
line_parseId_dict = {}
with open("../txt_res/en.txt") as file:
    i = 1   # start from 1
    for line in file:
        pairID = line.split(" ")[0]   # e.g. '1840'
        line_parseId_dict[pairID] = i    # line_parseId_dict[1840]=1
        i += 1

# pprint.pprint(line_parseId_dict, width=1)
# print(line_parseId_dict['152'])  => 825

pickle.dump(line_parseId_dict, open("../pickle_res/line_parseId_dict.p", "wb"))