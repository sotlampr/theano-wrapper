import numpy as np
import theano
from theano import tensor as T


class DummyClf:
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
    def __init__(self, n_in=100, n_out=10, **kwargs):
        self.X = T.matrix('X')
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


