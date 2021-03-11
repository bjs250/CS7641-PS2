import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import pickle
import matplotlib.pyplot as plt

filehandler = open('NNparams/GD_Model.obj', 'rb')
GD = pickle.load(filehandler)

filehandler = open('NNparams/RHC_Model.obj', 'rb')
RHC = pickle.load(filehandler)

filehandler = open('NNparams/SA_Model.obj', 'rb')
SA = pickle.load(filehandler)

filehandler = open('NNparams/GA_Model.obj', 'rb')
GA = pickle.load(filehandler)

if False:
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Iterations to convergence")
    plt.grid()

    m = 1250
    X = range(0, m)
    plt.plot(X, GD.fitness_curve[0:m], label="Backprop")
    plt.plot(X, RHC.fitness_curve[0:m]*-1, label="RHC")
    plt.plot(X, SA.fitness_curve[0:m]*-1, label="SA")
    plt.plot(X, GA.fitness_curve[0:m]*-1, label="GA")
    plt.legend(loc="lower right")

    plt.show()

if False:
    plt.xlabel("Weights")
    plt.ylabel("Values")
    plt.title("NN Weights")
    plt.grid()

    m = 272
    X = range(0, m)
    plt.plot(X, GD.fitted_weights[0:m], label="Backprop")
    plt.plot(X, RHC.fitted_weights[0:m]*-1, label="RHC")
    plt.plot(X, SA.fitted_weights[0:m]*-1, label="SA")
    # plt.plot(X, GA.fitted_weights[0:m]*-1, label="GA")
    plt.legend(loc="lower right")

    plt.show()

if True:
    X_train, y_train, X_test, y_test = preprocessing.preprocess()

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    NN = GA
    y_pred_train = NN.predict(X_train)
    y_pred_test = NN.predict(X_test)

    train_acc = metrics.accuracy_score(y_train, y_pred_train)
    test_acc = metrics.accuracy_score(y_test, y_pred_test)
    print("Train Accuracy: ", train_acc)
    print("Test Accuracy", test_acc)

    print(precision_recall_fscore_support(y_test, y_pred_test, average='weighted'))
    print(confusion_matrix(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))
