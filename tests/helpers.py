""" Helper classes and functions for testing theano-wrapper """
import numpy as np
import theano
from theano import tensor as T


def simple_reg(clf):
    """ L1 and L2 squared simple regularization """
    weights = (p for p in clf.params if p.name == 'W')

    L1 = T.sum([T.sum(abs(w)) for w in weights])
    L2_sqr = T.sum([T.sum(T.pow(w, 2)) for w in weights])

    return clf.cost + 0.001 * L1 + 0.001 * L2_sqr


class SimpleClf:
    """ Multi-class Logistic Regression.
    Attributes:
        X: theano input (from BaseLayer)
        y: theano output (from BaseLayer)
        W: theano weights (from BaseLayer)
        b: theano bias (from BaseLayer)
        params: list(W, b) (from BaseLayer)

        probas: (theano expression) Calculate probabilities for input X
        predict: (theano expression) Return the most probable class
        cost: (theano expression) Negative log-likelihood
        errors: Number of wrongly predicted samples

    """
    def __init__(self, n_in=100, n_out=10, X=None):
        self.X = T.matrix('X') if X is None else X
        self.y = T.ivector('y')
        _weights = np.zeros((n_in, n_out), dtype=theano.config.floatX)
        _bias = np.zeros(n_out,)

        self.W = theano.shared(_weights, name='W')
        self.b = theano.shared(_bias, name='b')
        self.params = [self.W, self.b]

        # symbolic expression for computing the matrix of probabilities
        self.probas = T.nnet.softmax(T.dot(self.X, self.W) + self.b)

        # symbolic expression for returning the most probable class
        self.predict = T.argmax(self.probas, axis=1)

        self.cost = -T.mean(
            T.log(self.probas)[T.arange(self.y.shape[0]), self.y])

        self.errors = T.mean(T.neq(self.predict, self.y))


class SimpleTrainer:
    """ Simple Trainer. Train a network for 30 epochs """
    def __init__(self, clf, reg=None):
        self.clf = clf
        self.X = clf.X
        self.y = clf.y
        self.cost = reg if reg else clf.cost

        self.grads = [T.grad(self.cost, p) for p in self.clf.params]
        self.updates = [(p, p - 0.001 * g)
                        for p, g in zip(self.clf.params, self.grads)]

    def fit(self, X, y):
        train_model = theano.function(inputs=[self.X, self.y],
                                      updates=self.updates)
        for _ in range(30):
            train_model(X, y)
