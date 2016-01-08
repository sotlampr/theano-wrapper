""" Various layers for machine learning applications
Classes:
    LinearRegression: simple linear regression
    LogisticRegression: simple logistic regression
"""
import numpy as np

import theano
from theano import tensor as T

from theano_wrapper.common import RandomBase


# BASE CLASSES ##############################################################
# pylint: disable=invalid-name
#   Names like X,y, X_train, y_train etc. are common in machine learning
#   tasks. For better readability and comprehension, disable pylint on
#   invalid names.
# pylint: disable=too-few-public-methods
#   Theano uses internal symbolic functions, it's ok for these classes
#   to have too few public methods
class BaseLayer:
    """ Base Class for all layers
    Attributes:
         X: (theano matrix) Theano symbolic input
         y: (theano vector) Theano symbolic output (optional)
         W: (theano arr(n_in, n_out)) Weights matrix
         b: (theano arr(n_out,)) Bias vector
         params: list(W, b) List containing the layer parameters W and b
    """
    def __init__(self, n_in, n_out, y=None, weights=None):
        """Arguements:
            n_in: (int) Number of input nodes
            n_out: (int) Number of output nodes
            y: (str, 'int' or 'float') Type of prediction vector (optional)
        """
        self.X = T.matrix('X')

        if y is None:
            # This is a hidden layer, no output vector
            pass
        elif isinstance(y, str):
            if y == 'int':
                self.y = T.ivector('y')
            elif y == 'float':
                self.y = T.fvector('y')
            else:
                # Handle the exception
                raise ValueError

        _weights, _bias = self.__init_weights_bias(weights, n_in, n_out)

        self.W = theano.shared(_weights, name='W')
        self.b = theano.shared(_bias, name='b')
        self.params = [self.W, self.b]

    @staticmethod
    def __init_weights_bias(weights, n_in, n_out):
        # Weights and bias initialization
        if weights is None:
            if n_out == 1:
                _weights = np.zeros((n_in), dtype=theano.config.floatX)
            else:
                _weights = np.zeros((n_in, n_out), dtype=theano.config.floatX)
        else:
            _weights = weights

        _bias = 0. if n_out == 1 else np.zeros(n_out,)

        return _weights, _bias


class HiddenLayer(RandomBase, BaseLayer):
    """ A Hidden layer
    Attributes:
        X: theano input (from BaseLayer)
        W: theano weights (from BaseLayer)
        b: theano bias (from BaseLayer)
        params: list(W, b) (from BaseLayer)

        activation: The activation function
        rng: numpy RandomState generator instance
        output: (theano expression) Expression to calculate layer activation
    """
    def __init__(self, n_in, n_out, activation=None, random=None, **kwargs):
        """ Parameters:
            n_in: (int) Number of input nodes.
            n_out: (int) Number of output nodes.
            activation (theano function) The activation function
            random: (int or numpy RandomState generator)
        """
        RandomBase.__init__(self, random)
        _size = (n_in,) if n_out == 1 else (n_in, n_out)
        weights = np.asarray(self._rng.uniform(low=-np.sqrt(6./(n_in+n_out)),
                                               high=np.sqrt(6./(n_in+n_out)),
                                               size=_size),
                             dtype=theano.config.floatX)

        BaseLayer.__init__(self, n_in, n_out, weights=weights, **kwargs)
        if activation:
            self.activation = activation
        else:
            self.activation = T.tanh

        self.output = self.activation(T.dot(self.X, self.W) + self.b)


# ESTIMATORS ################################################################
class LinearRegression(BaseLayer):
    """ Simple Linear Regression.
    Attributes:
        X: theano input (from BaseLayer)
        y: theano output (from BaseLayer)
        W: theano weights (from BaseLayer)
        b: theano bias (from BaseLayer)

        predict: (theano expression) Predict target value for input X
        cost: (theano expression) Mean squared error loss function
    """
    def __init__(self, *args, **kwargs):
        # Initialize Base:ayer
        super().__init__(*args, 'float', **kwargs)
        self.predict = T.dot(self.X, self.W) + self.b
        self.cost = T.sum(T.pow(self.predict-self.y, 2)) / (2*self.X.shape[0])


class LogisticRegression(BaseLayer):
    """ Multi-class Logistic Regression.
    Attributes:
        X: theano input (from BaseLayer)
        y: theano output (from BaseLayer)
        W: theano weights (from BaseLayer)
        b: theano bias (from BaseLayer)

        predict: (theano expression) Return the most probable class
        cost: (theano expression) Negative log-likelihood
        probas: (theano expression) Calculate probabilities for input X
        errors: Number of wrongly predicted samples

    """
    def __init__(self, *args, **kwargs):
        """ Parameters:
            n_in: (int) Number of input nodes.
            n_out: (int) Number of output nodes.
        """
        # Initialize BaseLayer
        super().__init__(*args, 'int', **kwargs)
        # symbolic expression for computing the matrix of probabilities
        self.probas = T.nnet.softmax(T.dot(self.X, self.W) + self.b)

        # symbolic expression for returning the most probable class
        self.predict = T.argmax(self.probas, axis=1)

        self.cost = -T.mean(
            T.log(self.probas)[T.arange(self.y.shape[0]), self.y])

        self.errors = T.mean(T.neq(self.predict, self.y))
# pylint: enable=invalid-name
# pylint: enable=too-few-public-methods
