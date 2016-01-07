""" Theano wrapper for your machine learning needs.
Available demos:
    Classification:
        1: Epoch-based Logistic Regression on the iris dataset.
    Regression:
        1: Epoch-based Linear Regression on the boston housing dataset.
"""
# Names like X,y, X_train, y_train etc. are common in machine learning
# tasks. For better readability and comprehension, disable pylint on
# invalid names.
# pylint: disable=invalid-name
import time
import sys

import numpy as np
from sklearn.datasets import fetch_mldata, load_boston, load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, mean_squared_error
import theano

from theano_wrapper.layers import (LogisticRegression, LinearRegression)
from theano_wrapper.trainers import EpochTrainer


RANDOM_STATE = 42
EQ_BAR = ''.join(['=' for i in range(50)])
HASH_BAR = ''.join(['#' for i in range(50)])


# HELPER UTILITIES
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


# DEMOS
def demo(choice=None):
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
                run_demo(choice)
            elif choice == '':
                print("\rPlease make a choice.")
            else:
                print("\rInvalid choice.")
    else:
        if isinstance(choice, str):
            if len(choice) == 2 and choice[0] in ('c', 'r'):
                try:
                    int(choice[1])
                    run_demo(choice[0], choice[1])
                except ValueError:
                    pass


def run_demo(task, example=None):
    """ Run a given demo interactive session or a specific example.
    Arguements:
        task: (str) 'c' or 'r' for Classification or Regression
        example: (str) number of the example to run
    """
    if task == 'r':
        regression_demos(example)
    elif task == 'c':
        classification_demos(example)
    else:
        return 1
    return 0


# Classification
def classification_demos(example=None):
    """ Run a classification demos interactive session or a specific example.
    Arguements:
        example: (str) number of example to run. if None, run the interactive
                       session
    """
    def run_example(ex):
        """ Run an examples. Any addition examples should be added here """
        if ex == '1':
            epoch_logreg()

    if example:
        run_example(example)

    else:
        while True:
            print("\t\t\tEnter a choice, [p] for a list of available "
                  "demos, or [b] "
                  "to go back.")
            choice = input().lower()
            if choice == 'b':
                return
            elif choice == 'p':
                print_classification()
            elif choice in ['1']:
                run_example(choice)
            else:
                print("Invalid choice.")


def print_classification():
    """ Print available classification demos """
    print("\nAvailable Classification demos:")
    print(HASH_BAR)
    print("==> 1:")
    print("\tEpoch-based Logistic Regression on the Iris Dataset.")
    print("")


def epoch_logreg():
    """ Epoch-based Logistic Regression on the iris dataset """
    print(EQ_BAR)
    print("Classification demo using Logistic Regression and an "
          "epoch-based trainer on the Iris dataset.")
    print(EQ_BAR)
    data, target_names = load_iris_data()
    X_train, X_test, y_train, y_test = data
    n_in = X_test.shape[1]
    n_out = len(np.unique(y_train))
    clf = LogisticRegression(n_in, n_out)
    trainer = EpochTrainer(clf, alpha=0.004, patience=12000, max_iter=200000,
                           imp_thresh=0.97, random=RANDOM_STATE, verbose=10)
    begin = time.time()
    trainer.fit(X_train, y_train)
    y_pred = trainer.predict(X_test)
    print("\n"+classification_report(y_test, y_pred,
                                     target_names=target_names))
    print("Took {:.1f} seconds\n".format(time.time()-begin))


# Regression
def regression_demos(example=None):
    """ Run a classification demos interactive session or a specific example.
    Arguements:
        example: (str) number of example to run. if None, run the interactive
                       session
    """
    def run_example(ex):
        """ Run an example. Any additional examples should be added here. """
        if ex == '1':
            epoch_linear()

    if example:
        run_example(example)
    else:
        while True:
            print("\t\t\tEnter a choice, [p] for a list of available demos, "
                  "or [b] to go back.")
            choice = input().lower()
            if choice == 'b':
                return
            elif choice == 'p':
                print_regression()
            elif choice in ['1']:
                run_example(choice)
            else:
                print("Invalid choice.")


def print_regression():
    """ Print available regression demos """
    print("\nAvailable Regression demos:")
    print(HASH_BAR)
    print("==> 1:")
    print("\tEpoch-based Linear Regression on the Boston")
    print("\thousing dataset.")
    print("")


def epoch_linear():
    """ Epoch-based Linear Regression on the Boston housing dataset. """
    print(EQ_BAR)
    print("Regression demo using Linear Regression and an "
          "epoch-based trainer on the Boston Housing dataset.")
    print(EQ_BAR)
    X_train, X_test, y_train, y_test = load_boston_data()
    n_in = X_test.shape[1]
    clf = LinearRegression(n_in, 1)
    trainer = EpochTrainer(clf, alpha=0.01, patience=50000, max_iter=100000,
                           imp_thresh=0.999, random=RANDOM_STATE, verbose=10)
    begin = time.time()
    trainer.fit(X_train, y_train)
    y_pred = trainer.predict(X_test)
    print("\nRMSE: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
    print("Took {:.1f} seconds\n".format(time.time()-begin))


if __name__ == "__main__":
    demo()
    sys.exit(0)
