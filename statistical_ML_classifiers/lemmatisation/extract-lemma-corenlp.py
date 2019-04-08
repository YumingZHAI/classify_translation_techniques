output = open ("ENlemme_from_corenlp.txt", "w")

with open("../txt_res/en.txt.conll") as file:
    for line in file:
        line=line.strip("\n")
        tab = line.split(" ")[1:]
        for item in tab:
            print(item.split("/")[2], file=output, end=" ")
        print("",file=output)

output.close()
