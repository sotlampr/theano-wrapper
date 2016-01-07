""" Various layers for machine learning applications """
import numpy as np
import theano
from theano import tensor as T


class BaseLayer:
    """ Base Class for all layers
    Attributes:
         X: (theano matrix) Theano symbolic input
         y: (theano vector) Theano symbolic output (optional)
         W: (theano arr(n_in, n_out)) Weights matrix
         b: (theano arr(n_out,)) Bias vector
         params: list(W, b) List containing the layer parameters W and b
    """
    def __init__(self, n_in, n_out, y=None):
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

        _weights, _bias = self.__init_weights_bias(n_in, n_out)

        self.W = theano.shared(_weights, name='W')
        self.b = theano.shared(_bias, name='b')
        self.params = [self.W, self.b]

    @staticmethod
    def __init_weights_bias(n_in, n_out):
        # Weights and bias initialization
        if n_out == 1:
            _weights = np.zeros((n_in), dtype=theano.config.floatX)
            _bias = 0.
        else:
            _weights = np.zeros((n_in, n_out), dtype=theano.config.floatX)
            _bias = np.zeros(n_out,)

        return _weights, _bias


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
    def __init__(self, n_in, n_out):
        # Initialize Base:ayer
        super().__init__(n_in, n_out, 'float')
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
    def __init__(self, n_in, n_out):
        """ Parameters:
            n_in: (int) Number of input nodes.
            n_out: (int) Number of output nodes.
        """
        # Initialize BaseLayer
        super(LogisticRegression, self).__init__(n_in, n_out, 'int')
        # symbolic expression for computing the matrix of probabilities
        self.probas = T.nnet.softmax(T.dot(self.X, self.W) + self.b)

        # symbolic expression for returning the most probable class
        self.predict = T.argmax(self.probas, axis=1)

        self.cost = -T.mean(
            T.log(self.probas)[T.arange(self.y.shape[0]), self.y])

        self.errors = T.mean(T.neq(self.predict, self.y))
