import pandas as pd
import sys
import os
from sklearn.model_selection import StratifiedKFold
from collections import Counter

"""author: Yuming Zhai"""

# split csv format data set to five folds
# input: 2class_balanced.csv or 5classes.csv

## You have to change here for the two datasets 
dataframe = pd.read_csv("2class_balanced.csv", names=['source', 'target', 'label'], sep='\t', engine='python', header=0)
# dataframe = pd.read_csv("5classes.csv", names=['source', 'target', 'label'], sep='\t', engine='python', header=0)
array = dataframe.values
X = array[:, 0:2]   # excluding end index
y = array[:, 2].astype('int')    # must put .astype('int'), otherwise error in spliting
# count the number of instances for each class
print(Counter(y))

skf = StratifiedKFold(n_splits=5)

## You have to change here for the two datasets 
name_dir = "./k-folds-normalized/cross_validation_2class_balanced"
# name_dir = "./k-folds-normalized/cross_validation_5class"

if not os.path.exists(name_dir):
    os.makedirs(name_dir)

j = 1
for train_index, test_index in skf.split(X, y):
    src_train = open(name_dir + "/source_train_" + str(j) + ".txt", "w")
    tgt_train = open(name_dir + "/target_train_" + str(j) + ".txt", "w")
    label_train = open(name_dir + "/label_train_" + str(j) + ".txt", "w")
    for index in train_index:
        print(X[index][0], file=src_train)
        print(X[index][1], file=tgt_train)
        print(y[index], file=label_train)

    src_test = open(name_dir + "/source_test_" + str(j) + ".txt", "w")
    tgt_test = open(name_dir + "/target_test_" + str(j) + ".txt", "w")
    label_test = open(name_dir + "/label_test_" + str(j) + ".txt", "w")
    for index in test_index:
        print(X[index][0], file=src_test)
        print(X[index][1], file=tgt_test)
        print(y[index], file=label_test)

    j += 1