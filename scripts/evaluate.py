import sklearn.metrics as metrics
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
import numpy as np
import itertools

def qual_test(data, num_evaluate):
    for i in range(num_evaluate):
        ll = data[i][0] - np.amin(data[i][0])
        print('Test ' + str(i) + '...')
        print(ll)
        print(data[i][1])


def get_ratio(data):
    ratios = []
    for test in data:
        ll = (test[0] - np.amin(test[0])).tolist()
        switches = test[1]
        if len(switches) == 0:
            continue
        switch_vals = []
        for val in switches:
            switch_vals.append(ll.pop(int(val)))
        print('Test...')
        print(np.mean(switch_vals))
        print(np.mean(ll))
        # ratios.append(np.mean(switch_vals) / float(np.mean(ll)))

    print(ratios)
    return

def guess_switches(data):
    precisions = []
    recalls = []
    for utterance in data:
        ll = utterance[0] - np.amin(utterance[0])
        ll_mean = np.mean(ll)
        ll_std = np.std(ll)
        guesses = []
        for i,frame_ll in enumerate(ll):
            if frame_ll >= ll_mean*1.1:
                guesses.append(i)
        if len(utterance[1]) == 0:
            continue
        if len(guesses) != 0:
            total_correct = 0
            for guess in guesses:
                in_correct = list(filter(lambda x: x >= guess - 2 and x <= guess + 2, utterance[1]))
                if len(in_correct) > 0:
                    total_correct += 1
            precisions.append(total_correct / float(len(guesses)))
            recalls.append(total_correct / float(len(utterance[1])))
        else:
            recalls.append(0)

    print('Precision: ' + str(np.mean(precisions)))
    print('Recall: ' + str(np.mean(recalls)))


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
