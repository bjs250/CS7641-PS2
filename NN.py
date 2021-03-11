import pickle
import time

import preprocessing

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from mlrose_hiive.runners import SARunner
from mlrose_hiive.algorithms import GeomDecay, ExpDecay
from mlrose_hiive.neural import NNClassifier, NeuralNetwork

X_train, y_train, X_test, y_test = preprocessing.preprocess()

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# random_hill_climb, genetic_alg

def run_with_alg(algorithm):

    if algorithm == "gradient_descent":
        NN = NeuralNetwork(
            hidden_nodes=[10, 8, 8],
            activation="relu",
            algorithm="gradient_descent",
            max_iters=1000,
            bias=True,
            is_classifier=True,
            learning_rate=0.001,
            early_stopping=True,
            clip_max=1e+10,
            max_attempts=100,
            random_state=1,
            curve=True
        )

        start = time.time()
        NN.fit(X_train, y_train)
        stop = time.time()
        # file_pi = open('NNparams/GD_Model.obj', 'wb')
        # pickle.dump(NN, file_pi)

        print(f"Training time: {stop - start}s")
        y_pred_train = NN.predict(X_train)
        y_pred_test = NN.predict(X_test)

        train_acc = metrics.accuracy_score(y_train, y_pred_train)
        test_acc = metrics.accuracy_score(y_test, y_pred_test)
        print("Train Accuracy: ", train_acc)
        print("Test Accuracy", test_acc)

        print(precision_recall_fscore_support(y_test, y_pred_test, average='weighted'))
        print(confusion_matrix(y_test, y_pred_test))
        print(classification_report(y_test, y_pred_test))

    if algorithm == "random_hill_climb":
        NN = NeuralNetwork(
            hidden_nodes=[10, 8, 8],
            activation="relu",
            algorithm="random_hill_climb",
            max_iters=1000,
            bias=True,
            is_classifier=True,
            learning_rate=1,
            early_stopping=False,
            clip_max=1e+10,
            restarts=15,
            max_attempts=100,
            random_state=1,
            curve=True
        )

        start = time.time()
        NN.fit(X_train, y_train)
        stop = time.time()
        # file_pi = open('NNparams/RHC_Model.obj', 'wb')
        # pickle.dump(NN, file_pi)

        print(f"Training time: {stop - start}s")
        y_pred_train = NN.predict(X_train)
        y_pred_test = NN.predict(X_test)

        train_acc = metrics.accuracy_score(y_train, y_pred_train)
        test_acc = metrics.accuracy_score(y_test, y_pred_test)
        print("Train Accuracy: ", train_acc)
        print("Test Accuracy", test_acc)

        print(precision_recall_fscore_support(y_test, y_pred_test, average='weighted'))
        print(confusion_matrix(y_test, y_pred_test))
        print(classification_report(y_test, y_pred_test))

    if algorithm == "simulated_annealing":
        NN = NeuralNetwork(
            hidden_nodes=[10, 8, 8],
            activation="relu",
            algorithm="simulated_annealing",
            max_iters=1000,
            bias=True,
            is_classifier=True,
            learning_rate=1.4,
            early_stopping=False,
            clip_max=1e+10,
            schedule=GeomDecay(),
            max_attempts=100,
            random_state=1,
            curve=True
        )

        start = time.time()
        NN.fit(X_train, y_train)
        stop = time.time()
        # file_pi = open('NNparams/SA_Model.obj', 'wb')
        # pickle.dump(NN, file_pi)

        print(f"Training time: {stop - start}s")
        y_pred_train = NN.predict(X_train)
        y_pred_test = NN.predict(X_test)

        train_acc = metrics.accuracy_score(y_train, y_pred_train)
        test_acc = metrics.accuracy_score(y_test, y_pred_test)
        print("Train Accuracy: ", train_acc)
        print("Test Accuracy", test_acc)

        print(precision_recall_fscore_support(y_test, y_pred_test, average='weighted'))
        print(confusion_matrix(y_test, y_pred_test))
        print(classification_report(y_test, y_pred_test))

    # learning_rates = [0.001, 0.01, 0.1, 1.0, 2.0]
    # mutation_probs = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    # pop_sizes = [10, 50, 100, 200]
    if algorithm == "genetic_alg":
        learning_rates = [1.0]
        mutation_probs = [0.05]
        pop_sizes = [200]
        d = {}
        for learning_rate in learning_rates:
            for mutation_prob in mutation_probs:
                for pop_size in pop_sizes:
                    NN = NeuralNetwork(
                        hidden_nodes=[10, 8, 8],
                        activation="relu",
                        algorithm="genetic_alg",
                        max_iters=1000,
                        bias=True,
                        is_classifier=True,
                        learning_rate=learning_rate,
                        early_stopping=False,
                        clip_max=1e+10,
                        pop_size=pop_size,
                        mutation_prob=mutation_prob,
                        max_attempts=100,
                        random_state=1,
                        curve=True
                    )

                    start = time.time()
                    NN.fit(X_train, y_train)
                    stop = time.time()
                    # file_pi = open('NNparams/GA_Model.obj', 'wb')
                    # pickle.dump(NN, file_pi)

                    print(f"Training time: {stop - start}s")
                    y_pred_train = NN.predict(X_train)
                    y_pred_test = NN.predict(X_test)

                    train_acc = metrics.accuracy_score(y_train, y_pred_train)
                    test_acc = metrics.accuracy_score(y_test, y_pred_test)
                    print("Train Accuracy: ", train_acc)
                    print("Test Accuracy", test_acc)

                    print(precision_recall_fscore_support(y_test, y_pred_test, average='weighted'))
                    print(confusion_matrix(y_test, y_pred_test))
                    print(classification_report(y_test, y_pred_test))
                    d["%s-%s-%s-%s".format("GA", learning_rate, mutation_prob, pop_size)] = test_acc


run_with_alg("genetic_alg")
