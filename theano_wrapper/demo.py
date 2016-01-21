""" Theano wrapper for your machine learning needs.
Available demos:
    Classification:
        1: Epoch-based Logistic Regression on the iris dataset.
        2: Logistic Regression with Stohastic Gradient Descent
    Regression:
        1: Epoch-based Linear Regression on the boston housing dataset.
        2: Linear Regression with Stohastic Gradient Descent
"""
# Names like X,y, X_train, y_train etc. are common in machine learning
# tasks. For better readability and comprehension, disable pylint on
# invalid names.
# pylint: disable=invalid-name
import time
import sys

import numpy as np
from sklearn.datasets import (fetch_mldata, load_boston, load_iris,
                              load_linnerud)
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, mean_squared_error
import theano

from theano_wrapper.layers import (LogisticRegression, LinearRegression,
                                   MultiLayerPerceptron, MultiLayerRegression)
from theano_wrapper.trainers import EpochTrainer, SGDTrainer


RANDOM_STATE = 42
EQ_BAR = ''.join(['=' for i in range(50)])
HASH_BAR = ''.join(['#' for i in range(50)])


# HELPER UTILITIES ###########################################################
def load_mnist_data():
    """ Load the mnist handwritten digits using scikit-learn """
    mnist = fetch_mldata('MNIST Original')
    return train_test_split(
        MinMaxScaler().fit_transform(mnist.data.astype(theano.config.floatX)),
        mnist.target.astype(np.int32), test_size=0.25,
        random_state=RANDOM_STATE)


def load_boston_data():
    """ Load the boston house prices dataset using scikit-learn """
    boston = load_boston()
    return train_test_split(
        MinMaxScaler().fit_transform(boston.data.astype(theano.config.floatX)),
        boston.target.astype(theano.config.floatX), test_size=0.25,
        random_state=RANDOM_STATE)


def load_iris_data():
    """ Load the iris dataset using scikit-learn """
    iris = load_iris()
    return train_test_split(
        MinMaxScaler().fit_transform(iris.data.astype(theano.config.floatX)),
        iris.target.astype(np.int32), test_size=0.25,
        random_state=RANDOM_STATE), iris.target_names


def load_linnerud_data():
    """ Load the linnerud dataset using scikit-learn """
    linnerud = load_linnerud()
    return train_test_split(
        MinMaxScaler().fit_transform(
            linnerud.data.astype(theano.config.floatX)),
        linnerud.target.astype(theano.config.floatX), test_size=0.25,
        random_state=RANDOM_STATE)


# INTERACTIVE MODE ###########################################################
def demo(choice=None, __test=False):
    """ Interactive Demo.
    Usage:
        demo(), and follow on-screen instructions
        OR
        demo(arg), where arg is the desired demo in the form "<c,r><0-9>"
                   and the first letter representing the task
                   (r for regression, c for classification) and the last
                   integer the n-th example.
                   ex. demo('c2')
    """
    if not choice:
        print("Hello, please make a demo choice:")
        while True:
            print("\t\t\t[r] for Regression, [c] for Classificaton, "
                  "[q] for quit")
            choice = input().lower()
            if choice == 'q':
                break
            elif choice in ('c', 'r'):
                run_demo(choice, __test=__test)
            elif choice == '':
                print("\rPlease make a choice.")
            else:
                print("\rInvalid choice.")
    else:
        if isinstance(choice, str):
            if len(choice) == 2 and choice[0] in ('c', 'r'):
                try:
                    int(choice[1])
                    run_demo(choice[0], choice[1], __test)
                except ValueError:
                    pass


def run_demo(task, example=None, __test=False):
    """ Run a given demo interactive session or a specific example.
    Arguements:
        task: (str) 'c' or 'r' for Classification or Regression
        example: (str) number of the example to run
    """
    if task == 'r':
        regression_demos(example, __test)
    elif task == 'c':
        classification_demos(example, __test)
    else:
        return 1
    return 0


# HELPER FUNCTIONS ###########################################################
def classification_demos(example=None, __test=False):
    """ Run a classification demos interactive session or a specific example.
    Arguements:
        example: (str) number of example to run. if None, run the interactive
                       session
    """
    def run_example(ex, __test):
        """ Run an examples. Any addition examples should be added here """
        examples = {
            '1': epoch_logreg,
            '2': sgd_logreg,
            '3': mnist_epoch_logreg,
            '4': mnist_sgd_logreg,
            '5': mnist_mlp}
        examples[ex](__test)

    if example:
        run_example(example, __test)

    else:
        while True:
            print("\t\t\tEnter a choice, [p] for a list of available "
                  "demos, or [b] "
                  "to go back.")
            choice = input().lower()
            if choice == 'b':
                return
            elif choice == 'p':
                print_available('clf')
            elif choice in ['1', '2', '3', '4', '5']:
                run_example(choice, False)
            else:
                print("Invalid choice.")


def regression_demos(example=None, __test=False):
    """ Run a classification demos interactive session or a specific example.
    Arguements:
        example: (str) number of example to run. if None, run the interactive
                       session
    """
    def run_example(ex, __test):
        """ Run an example. Any additional examples should be added here. """
        examples = {
            '1': epoch_linear,
            '2': sgd_linear,
            '3': linnerud_linear_sgd,
            '4': linnerud_mlr}
        examples[ex](__test)

    if example:
        run_example(example, __test)
    else:
        while True:
            print("\t\t\tEnter a choice, [p] for a list of available demos, "
                  "or [b] to go back.")
            choice = input().lower()
            if choice == 'b':
                return
            elif choice == 'p':
                print_available('reg')
            elif choice in ['1', '2', '3', '4']:
                run_example(choice, False)
            else:
                print("Invalid choice.")


def print_available(wat):
    """ Print available classification or regression demos """
    if wat == 'clf':
        # Print available classification demos
        print("\nAvailable Classification demos:")
        print(HASH_BAR)
        print("==> 1:")
        print("\tEpoch-based Logistic Regression on the Iris Dataset.")
        print("==> 2:")
        print("\tLogistic Regression with Stohastic Gradient Descent "
              "on the Iris dataset.")
        print("==> 3:")
        print("\tEpoch-based Logistic Regression on the MNIST Dataset.")
        print("\t\t*It has many more samples than the Iris dataset, so it ")
        print("\t\t is a good example of why we need Stohastic "
              "Gradient Descent")
        print("==> 4:")
        print("\tLogistic Regression with Stohastic Gradient Descent "
              "on the MNIST dataset.")
        print("==> 5:")
        print("\tSingle layer multilayer perceptron on the MNIST dataset.")
        print("")

    elif wat == 'reg':
        # Print available regression demos
        print("\nAvailable Regression demos:")
        print(HASH_BAR)
        print("==> 1:")
        print("\tEpoch-based Linear Regression on the Boston")
        print("\thousing dataset.")
        print("==> 2:")
        print("\tLinear Regression with Stohastic Gradient Descent on the "
              "Boston housing dataset.")
        print("==> 3:")
        print("\tLinear Regression with Stohastic Gradient Descent on the "
              "linnerud multivariate dataset.")
        print("==> 4:")
        print("\tSingle layer multilayer regresson on the "
              "linnerud multivariate dataset.")
        print("")
    else:
        print_available('clf')
        print_available('reg')


def epoch_logreg(__test):
    """ Epoch-based Logistic Regression on the iris dataset """
    print(EQ_BAR)
    print("Classification demo using Logistic Regression and an "
          "epoch-based trainer on the Iris dataset.")
    print(EQ_BAR)
    max_iter = 1 if __test else 200000
    data, target_names = load_iris_data()
    X_train, X_test, y_train, y_test = data
    n_in = X_test.shape[1]
    n_out = len(np.unique(y_train))
    clf = LogisticRegression(n_in, n_out)
    trainer = EpochTrainer(clf, alpha=0.004, patience=12000, max_iter=max_iter,
                           imp_thresh=0.986, random=RANDOM_STATE, verbose=10)
    begin = time.time()
    trainer.fit(X_train, y_train)
    y_pred = trainer.predict(X_test)
    print("\n"+classification_report(y_test, y_pred,
                                     target_names=target_names))
    print("Took {:.1f} seconds\n".format(time.time()-begin))


def sgd_logreg(__test):
    """ Stohastic Gradient Descent Logistic Regression on the iris dataset """
    print(EQ_BAR)
    print("Classification demo using Logistic Regression and Stohastic "
          "Gradient Descent on the Iris dataset.")
    print(EQ_BAR)
    max_iter = 1 if __test else 200000
    data, target_names = load_iris_data()
    X_train, X_test, y_train, y_test = data
    n_in = X_test.shape[1]
    n_out = len(np.unique(y_train))
    clf = LogisticRegression(n_in, n_out)
    trainer = SGDTrainer(clf, batch_size=2, alpha=0.03, patience=10000,
                         max_iter=max_iter, imp_thresh=0.999,
                         random=RANDOM_STATE, verbose=3)
    begin = time.time()
    trainer.fit(X_train, y_train)
    y_pred = trainer.predict(X_test)
    print("\n"+classification_report(y_test, y_pred,
                                     target_names=target_names))
    print("Took {:.1f} seconds\n".format(time.time()-begin))


def mnist_epoch_logreg(__test):
    """ Epoch-based Logistic Regression on the MNIST dataset """
    print(EQ_BAR)
    print("Classification demo using Logistic Regression and an "
          "epoch-based trainer on the MNIST dataset.")
    print(EQ_BAR)
    max_iter = 1 if __test else 200000
    X_train, X_test, y_train, y_test = load_mnist_data()
    n_in = X_test.shape[1]
    n_out = len(np.unique(y_train))
    clf = LogisticRegression(n_in, n_out)
    trainer = EpochTrainer(clf, alpha=0.9, patience=50, max_iter=max_iter,
                           imp_thresh=0.95, random=RANDOM_STATE, verbose=1)
    begin = time.time()
    trainer.fit(X_train, y_train)
    y_pred = trainer.predict(X_test)
    print("\n"+classification_report(y_test, y_pred))
    print("Took {:.1f} seconds\n".format(time.time()-begin))


def mnist_sgd_logreg(__test):
    """ Stohastic Gradient Descent Logistic Regression on the MNIST digits
    database
    """
    print(EQ_BAR)
    print("Classification demo using Logistic Regression and Stohastic "
          "Gradient Descent on the MNIST digits dataset.")
    print(EQ_BAR)
    max_iter = 1 if __test else 100000
    X_train, X_test, y_train, y_test = load_mnist_data()
    n_in = X_test.shape[1]
    n_out = len(np.unique(y_train))
    clf = LogisticRegression(n_in, n_out)
    trainer = SGDTrainer(clf, batch_size=100, alpha=0.5, patience=1000,
                         max_iter=max_iter,
                         imp_thresh=0.99, random=RANDOM_STATE, verbose=3)
    begin = time.time()
    trainer.fit(X_train, y_train)
    y_pred = trainer.predict(X_test)
    print("\n"+classification_report(y_test, y_pred))
    print("Took {:.1f} seconds\n".format(time.time()-begin))


def mnist_mlp(__test):
    """ Stohastic Gradient Descent Multilayer Perceptron on the MNIST digits
    database
    """
    print(EQ_BAR)
    print("Classification demo using a Multilayer Perceptron and Stohastic "
          "Gradient Descent on the MNIST digits dataset.")
    print(EQ_BAR)
    max_iter = 1 if __test else 100000
    X_train, X_test, y_train, y_test = load_mnist_data()
    n_in = X_test.shape[1]
    n_out = len(np.unique(y_train))
    clf = MultiLayerPerceptron(n_in, int(n_in/8), n_out)
    trainer = SGDTrainer(clf, batch_size=100, alpha=0.7, patience=500,
                         max_iter=max_iter,
                         imp_thresh=0.9, random=RANDOM_STATE, verbose=3)
    begin = time.time()
    trainer.fit(X_train, y_train)
    y_pred = trainer.predict(X_test)
    print("\n"+classification_report(y_test, y_pred))
    print("Took {:.1f} seconds\n".format(time.time()-begin))


def epoch_linear(__test):
    """ Epoch-based Linear Regression on the Boston housing dataset. """
    print(EQ_BAR)
    print("Regression demo using Linear Regression and an "
          "epoch-based trainer on the Boston Housing dataset.")
    print(EQ_BAR)
    max_iter = 1 if __test else 100000
    X_train, X_test, y_train, y_test = load_boston_data()
    n_in = X_test.shape[1]
    clf = LinearRegression(n_in, 1)
    trainer = EpochTrainer(clf, alpha=0.1, patience=500, max_iter=max_iter,
                           imp_thresh=1, random=RANDOM_STATE, verbose=10)
    begin = time.time()
    trainer.fit(X_train, y_train)
    y_pred = trainer.predict(X_test)
    print("\nRMSE: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
    print("Took {:.1f} seconds\n".format(time.time()-begin))


def sgd_linear(__test):
    """ Stohastic Gradient Descent Linear Regression on the
    Boston housing dataset.
    """
    print(EQ_BAR)
    print("Regression demo using Linear Regression and a "
          "Stohastic Gradient Descent trainer on the Boston Housing dataset.")
    print(EQ_BAR)
    max_iter = 1 if __test else 100000
    X_train, X_test, y_train, y_test = load_boston_data()
    n_in = X_test.shape[1]
    clf = LinearRegression(n_in, 1)
    trainer = SGDTrainer(clf, batch_size=10, alpha=0.08, patience=200,
                         imp_thresh=0.999999, max_iter=max_iter,
                         random=RANDOM_STATE, verbose=10)
    begin = time.time()
    trainer.fit(X_train, y_train)
    y_pred = trainer.predict(X_test)
    print("\nRMSE: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
    print("Took {:.1f} seconds\n".format(time.time()-begin))


def linnerud_linear_sgd(__test):
    """ Stohastic Gradient Descent Linear Regression on the
    linnerud multivariate dataset.
    """
    print(EQ_BAR)
    print("Regression demo using Linear Regression and a "
          "Stohastic Gradient Descent trainer on the linnerud "
          "multivariate dataset.")
    print(EQ_BAR)
    max_iter = 1 if __test else 100000
    X_train, X_test, y_train, y_test = load_linnerud_data()
    n_in = X_test.shape[1]
    n_out = y_test.shape[1]
    clf = LinearRegression(n_in, n_out)
    trainer = SGDTrainer(clf, batch_size=3, alpha=0.0005, patience=100,
                         imp_thresh=1, max_iter=max_iter,
                         random=RANDOM_STATE, verbose=3)
    begin = time.time()
    trainer.fit(X_train, y_train)
    y_pred = trainer.predict(X_test)
    print("\nRMSE: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
    print("Took {:.1f} seconds\n".format(time.time()-begin))


def linnerud_mlr(__test):
    """ Stohastic Gradient Descent Multilayer Regression on the
    linnerud multivariate dataset.
    """
    print(EQ_BAR)
    print("Regression demo using Multilayer Regression and a "
          "Stohastic Gradient Descent trainer on the linnerud "
          "multivariate dataset.")
    print(EQ_BAR)
    max_iter = 1 if __test else 100000
    X_train, X_test, y_train, y_test = load_linnerud_data()
    n_in = X_test.shape[1]
    n_out = y_test.shape[1]
    clf = MultiLayerRegression(n_in, 10, n_out)
    trainer = SGDTrainer(clf, batch_size=1, alpha=0.01, patience=1000,
                         imp_thresh=0.99999, max_iter=max_iter,
                         random=RANDOM_STATE, verbose=1)
    begin = time.time()
    trainer.fit(X_train, y_train)
    y_pred = trainer.predict(X_test)
    print("\nRMSE: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
    print("Took {:.1f} seconds\n".format(time.time()-begin))

if __name__ == "__main__":
    demo()
    sys.exit(0)
