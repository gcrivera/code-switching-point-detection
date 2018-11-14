import sklearn.metrics as metrics

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
