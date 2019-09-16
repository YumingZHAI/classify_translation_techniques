from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, f1_score
import pdb
import numpy as np
import sys

name = sys.argv[1]

fold1_pred = './' + name + '_preds_and_labels/fold_1/models_best_all_preds.txt'
fold1_label = './' + name + '_preds_and_labels/fold_1/models_best_all_labels.txt'

fold2_pred = './' + name + '_preds_and_labels/fold_2/models_best_all_preds.txt'
fold2_label = './' + name + '_preds_and_labels/fold_2/models_best_all_labels.txt'

fold3_pred = './' + name + '_preds_and_labels/fold_3/models_best_all_preds.txt'
fold3_label = './' + name + '_preds_and_labels/fold_3/models_best_all_labels.txt'

fold4_pred = './' + name + '_preds_and_labels/fold_4/models_best_all_preds.txt'
fold4_label = './' + name + '_preds_and_labels/fold_4/models_best_all_labels.txt'

fold5_pred = './' + name + '_preds_and_labels/fold_5/models_best_all_preds.txt'
fold5_label = './' + name + '_preds_and_labels/fold_5/models_best_all_labels.txt'

all_pred = []
all_labels = []
f1_array = []

preds_1 = []
labels_1 = []
with open(fold1_pred) as f:
    for y in f.read().splitlines():
        all_pred.append(int(y))
        preds_1.append(int(y))
with open(fold1_label) as f:
    for y in f.read().splitlines():
        all_labels.append(int(y))
        labels_1.append(int(y))
f1_fold1 = precision_recall_fscore_support(labels_1, preds_1, average=None)[2]
f1_array.append(f1_fold1)

preds_2 = []
labels_2 = []
with open(fold2_pred) as f:
    for y in f.read().splitlines():
        all_pred.append(int(y))
        preds_2.append(int(y))
with open(fold2_label) as f:
    for y in f.read().splitlines():
        all_labels.append(int(y))
        labels_2.append(int(y))
f1_fold2 = precision_recall_fscore_support(labels_2, preds_2, average=None)[2]
f1_array.append(f1_fold2)

preds_3 = []
labels_3= []
with open(fold3_pred) as f:
    for y in f.read().splitlines():
        all_pred.append(int(y))
        preds_3.append(int(y))
with open(fold3_label) as f:
    for y in f.read().splitlines():
        all_labels.append(int(y))
        labels_3.append(int(y))
f1_fold3 = precision_recall_fscore_support(labels_3, preds_3, average=None)[2]
f1_array.append(f1_fold3)

preds_4 = []
labels_4 = []
with open(fold4_pred) as f:
    for y in f.read().splitlines():
        all_pred.append(int(y))
        preds_4.append(int(y))
with open(fold4_label) as f:
    for y in f.read().splitlines():
        all_labels.append(int(y))
        labels_4.append(int(y))
f1_fold4 = precision_recall_fscore_support(labels_4, preds_4, average=None)[2]
f1_array.append(f1_fold4)

preds_5 = []
labels_5 = []
with open(fold5_pred) as f:
    for y in f.read().splitlines():
        all_pred.append(int(y))
        preds_5.append(int(y))
with open(fold5_label) as f:
    for y in f.read().splitlines():
        all_labels.append(int(y))
        labels_5.append(int(y))
f1_fold5 = precision_recall_fscore_support(labels_5, preds_5, average=None)[2]
f1_array.append(f1_fold5)

# average = None: the scores for each class are returned.
# Otherwise, this determines the type of averaging performed on the data
F1_score = precision_recall_fscore_support(all_labels, all_pred, average=None)[2]
print('overall F1-score by summing predictions: ',F1_score)

avg_f1score = np.mean(f1_array, axis=0)
print('mean f1 per class on five folds: ', avg_f1score)

accuracy1 = accuracy_score(labels_1, preds_1)
accuracy2 = accuracy_score(labels_2, preds_2)
accuracy3 = accuracy_score(labels_3, preds_3)
accuracy4 = accuracy_score(labels_4, preds_4)
accuracy5 = accuracy_score(labels_5, preds_5)
accuracy_list = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5]
accuracy_mean = np.mean(accuracy_list)
print('mean accuracy over five fold: ', accuracy_mean)

print('micro f1: {}'.format(f1_score(all_labels, all_pred, average='micro')))
print('macro f1: {}'.format(f1_score(all_labels, all_pred, average='macro')))