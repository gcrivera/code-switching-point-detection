import sklearn.metrics as metrics
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
import numpy as np
import itertools

def get_predictions(scores):
    Y = []
    Y_pred = []

    switch_lls = scores['switch']
    for i in range(len(switch_lls)):
        Y.append(0)
        switch_ll = switch_lls[i][0]
        non_switch_ll = switch_lls[i][1]
        if switch_ll >= non_switch_ll:
            Y_pred.append(0)
        else:
            Y_pred.append(1)

    non_switch_lls = scores['non_switch']
    for i in range(len(non_switch_lls)):
        Y.append(1)
        switch_ll = non_switch_lls[i][0]
        non_switch_ll = non_switch_lls[i][1]
        if non_switch_ll >= switch_ll:
            Y_pred.append(1)
        else:
            Y_pred.append(0)

    return Y,Y_pred

def evaluate(Y, Y_pred):
    # calculate accuracy
    accuracy = metrics.accuracy_score(Y, Y_pred)
    print('Accuracy:')
    print('\t' + str(accuracy))

    # calculate weighted F1 Scores
    f1_score = metrics.f1_score(Y, Y_pred, labels=[0, 1], average='weighted')
    print('Weighted F1 Score:')
    print('\t' + str(f1_score))

# def confusion_matrix(Y, Y_pred):
#     cm = metrics.confusion_matrix(Y, Y_pred)
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     classes = [0, 1]
#
#     plt.figure()
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title('Confusion matrix')
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()
#     plt.show()
