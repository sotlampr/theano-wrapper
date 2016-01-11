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

    A base class providing input, output weights and bias.

    Attributes:
        X (theano matrix): Theano symbolic input
        y (theano vector): Theano symbolic output (optional)
        W (theano arr(n_in, n_out)): Weights matrix
        b (theano arr(n_out,)): Bias vector
        params (list(W, b)): List containing the layer parameters W and b
    """
    def __init__(self, n_in, n_out, y=None, X=None, weights=None):
        """Arguements:
            n_in (int): Number of input nodes
            n_out (int): Number of output nodes
            y (str, 'int' or 'float'): Type of prediction vector (optional)
        """
        self.X = T.matrix('X') if X is None else X

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


class MultiLayerBase(RandomBase):
    """ A Multi layer network

    Attrs:
        n_in (int): number of input nodes
        n_hidden (int or list(int)): if int: a single layer network of n_hidden
            nodes. If list: a multi layer network consisting of
            m = len(n_hidden) layers and nodes for each layer given
            by n_hidden[m]
        n_out (int): number of output layers
        out_layer (estimator class): type of the final layer
        activation (function or list of functions): In accordance with
            n_hidden. If network is single layer must be a function, if multi
            layer can be either a function or a list of functions.
    """
    def __init__(self, n_in, n_hidden, n_out, out_layer, activation=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = []
        if isinstance(n_hidden, int):
            if isinstance(activation, list):
                # handle the exception
                raise TypeError
            n_prev = n_hidden
            self.layers.append(HiddenLayer(n_in, n_hidden,
                                           activation, self._rng))
        elif isinstance(n_hidden, list):
            if not isinstance(activation, list):
                temp = activation
                activation = [temp for i in range(len(n_hidden))]
            self.layers.append(HiddenLayer(n_in, n_hidden[0],
                                           activation[0], self._rng))
            n_prev = n_hidden[0]
            for i, n_layer in enumerate(n_hidden):
                if i == 0:
                    continue
                self.layers.append(HiddenLayer(n_prev, n_layer, activation[i],
                                               self._rng,
                                               X=self.layers[i-1].output))
                n_prev = n_layer
        self.output_layer = out_layer(n_prev, n_out, X=self.layers[-1].output)

        self.X = self.layers[0].X
        self.y = self.output_layer.y
        self.cost = self.output_layer.cost
        self.predict = self.output_layer.predict
        self.params = [p for l in self.layers for p in l.params]


# ESTIMATORS ################################################################
class LinearRegression(BaseLayer):
    """ Simple Linear Regression.
    Linear regression is a linear predictor modeling the relationship
    between a scalar dependent variable :math:`y` and one or more explanatory
    variables denoted :math:`D` from an input sample :math:`X`. The target
    value is given by the formula:

        .. math::

           y = \sum_{i=0}^{|\mathcal{D}|} (W_d \cdot X_d) +  b

    Args:
        n_in (int): Number of input nodes
        n_out (int): Number of output nodes

    Attributes:

        X (theano variable): Symbolic input.
        y (theano variable): Symbolic output.
        W (theano variable): Weights matrix, shape=(n_in, n_out).
        b (theano variable): Bias vector, shape=(n_out,).


        predict (theano expression): Predict target value for input X.
        cost (theano expression): Mean squared error loss function.
    """
    def __init__(self, n_in, n_out):
        # Initialize BaseLayer and theano symbolic functions
        super().__init__(n_in, n_out, 'float')
        self.predict = T.dot(self.X, self.W) + self.b
        self.cost = T.sum(T.pow(self.predict-self.y, 2)) / (2*self.X.shape[0])


class LogisticRegression(BaseLayer):
    r""" Multi-class Logistic Regression.

    Logistic regression is a probabilistic, linear classifier. It is
    parametrized by a weight matrix :math:`W` and a bias vector :math:`b`.
    Classification is done by projecting an input vector onto a set of
    hyperplanes, each of which corresponds to a class. The distance from the
    input to a hyperplane reflects the probability that the input is a member
    of the corresponding class.


    .. math::
          P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                        &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


    The model's prediction :math:`y_{pred}` is the class whose probability
    is maximal, specifically:

    .. math::
        y_{pred} = {\rm argmax}_i P(Y=i|x,W,b)


    Args:
        n_in (int): Number of input nodes
        n_out (int): Number of output nodes

    Attributes:

        X (theano variable): Symbolic input.
        y (theano variable): Symbolic output.
        W (theano variable): Weights matrix, shape=(n_in, n_out).
        b (theano variable): Bias vector, shape=(n_out,).

        predict (theano expression): Return the most probable class
            (the probability function as described above).
        cost (theano expression): Negative log-likelihood if we define the
            likelihood :math:`\cal{L}` and loss :math:`\ell`:

            .. math::

               \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
                 \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
               \ell (\theta=\{W,b\}, \mathcal{D}) =
                  - \mathcal{L} (\theta=\{W,b\}, \mathcal{D})
        probas (theano expression): Calculate probabilities for input X.
        errors (theano expression): Number of wrongly predicted samples.

    """
    def __init__(self, n_in, n_out):
        """ Initialize BaseLayer and the following theano symbolic
        functions:
            probas: Class propabilities
            predict: Return the class with the greatest probability
            cost: Negative log-likelihood
            erros: Return count of errors
        """
        # Initialize BaseLayer
        super().__init__(n_in, n_out, 'int')
        # symbolic expression for computing the matrix of probabilities
        self.probas = T.nnet.softmax(T.dot(self.X, self.W) + self.b)

        # symbolic expression for returning the most probable class
        self.predict = T.argmax(self.probas, axis=1)

        self.cost = -T.mean(
            T.log(self.probas)[T.arange(self.y.shape[0]), self.y])

        self.errors = T.mean(T.neq(self.predict, self.y))
# pylint: enable=invalid-name
# pylint: enable=too-few-public-methods
