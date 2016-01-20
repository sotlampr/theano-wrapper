""" Various layers for machine learning applications
Classes:
    LinearRegression: simple linear regression
    LogisticRegression: simple logistic regression
"""
import numpy as np

import theano
from theano import tensor as T

from theano_wrapper.common import RandomBase
from theano_wrapper.trainers import EpochTrainer, SGDTrainer


# pylint: disable=invalid-name
#     Names like X,y, X_train, y_train etc. are common in machine learning
#     tasks. For better readability and comprehension, disable pylint on
#     invalid names.
# pylint: disable=too-few-public-methods
#     Theano uses internal symbolic functions, it's ok for these classes
#     to have too few public methods
# BASE CLASSES ###############################################################
# pylint: disable=too-many-arguments
#   Seems ok for these base classes to have a couple of more arguements.
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
    def __init__(self, n_in, n_out, y=None, X=None, weights=None, bias=None):
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
                self.y = T.fvector('y') if n_out == 1 else T.fmatrix('y')
            else:
                # Handle the exception
                raise ValueError
        else:
            self.y = y

        _weights, _bias = self.__init_weights_bias(weights, bias, n_in, n_out)

        self.W = theano.shared(_weights, name='W')
        self.b = theano.shared(_bias, name='b')
        self.params = [self.W, self.b]

    @staticmethod
    def __init_weights_bias(weights, bias, n_in, n_out):
        # Weights and bias initialization
        if n_out == 1:
            if weights is None:
                weights = np.zeros((n_in), dtype=theano.config.floatX)
            if bias is None:
                bias = .0
        else:
            if weights is None:
                weights = np.zeros((n_in, n_out), dtype=theano.config.floatX)
            if bias is None:
                bias = np.zeros(n_out, dtype=theano.config.floatX)

        return weights, bias


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
        self.layers.append(out_layer(n_prev, n_out, self.layers[-1].output))

        self.X = self.layers[0].X
        self.y = self.layers[-1].y
        self.cost = self.layers[-1].cost
        self.predict = self.layers[-1].predict
        self.params = [p for l in self.layers for p in l.params]


class BaseEstimator:
    trainer_aliases = {
        'epoch': EpochTrainer,
        'sgd': SGDTrainer}

    def __init__(self):
        self.trainer = EpochTrainer(self)

    def _init_trainer(self, alias, **kwargs):
        if alias not in self.trainer_aliases.keys():
            # handle exception
            raise KeyError
        self.trainer = self.trainer_aliases[alias](self, **kwargs)

    def fit(self, X, y, trainer=None, **kwargs):
        if trainer:
            if isinstance(trainer, str):
                self._init_trainer(trainer, **kwargs)
            else:
                self.trainer = trainer
        else:
            if not hasattr(self, 'trainer'):
                if X.shape[0] < 5000:
                    self._init_trainer('epoch', **kwargs)
                else:
                    self._init_trainer('sgd', **kwargs)

        return self.trainer.fit(X, y)

    def predict(self, X):
        return self.trainer.predict(X)


# pylint: enable=too-many-arguments
# ACTIVATION FUNCTIONS #######################################################
def relu(value):
    """ Rectified linear unit activation function """
    return theano.tensor.switch(value < 0, 0, value)


# ESTIMATORS #################################################################
class LinearRegression(BaseLayer, BaseEstimator):
    r""" Simple Linear Regression.
    Linear regression is a linear predictor modeling the relationship
    between a scalar dependent variable :math:`y` and one or more explanatory
    variables denoted :math:`D` from an input sample :math:`X`. The target
    value is given by the formula:

        .. math::

           y = \sum_{i=0}^{|\mathcal{D}|} (W_i \cdot X_i) +  b

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
    def __init__(self, n_in, n_out, *args, **kwargs):
        # Initialize BaseLayer and theano symbolic functions
        super().__init__(n_in, n_out, 'float', *args, **kwargs)
        self.predict = T.dot(self.X, self.W) + self.b
        self.cost = T.sum(T.pow(self.predict-self.y, 2)) / (2*self.X.shape[0])


class LogisticRegression(BaseLayer, BaseEstimator):
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
    """
    def __init__(self, n_in, n_out, *args, **kwargs):
        """ Initialize BaseLayer and the following theano symbolic
        functions:
            probas: Class propabilities
            predict: Return the class with the greatest probability
            cost: Negative log-likelihood
            erros: Return count of errors
        """
        # Initialize BaseLayer
        super().__init__(n_in, n_out, 'int', *args, **kwargs)
        # symbolic expression for computing the matrix of probabilities
        self.probas = T.nnet.softmax(T.dot(self.X, self.W) + self.b)

        # symbolic expression for returning the most probable class
        self.predict = T.argmax(self.probas, axis=1)

        self.cost = -T.mean(
            T.log(self.probas)[T.arange(self.y.shape[0]), self.y])


class MultiLayerRegression(MultiLayerBase, BaseEstimator):
    r""" Multilayer Regression.

    An MLP can be viewed as a linear regression predictor where the input
    is first transformed using a transformation :math:`\Phi`. This
    transformation projects the input data into a more sparse or dense space.
    This intermediate layer is referred to as a **hidden layer**.  Formally,
    a one-hidden-layer MLP is a function :math:`f: R^D \rightarrow R^L`,
    where :math:`D` is the size of input vector :math:`x` and :math:`L` is
    the size of the output vector :math:`f(x)`, such that, in matrix notation:
        .. math::

                f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

    with bias vectors :math:`b^{(1)}`, :math:`b^{(2)}`; weight matrices
    :math:`W^{(1)}`, :math:`W^{(2)}` and activation functions :math:`G` and
    :math:`s`. The vector :math:`h(x) = \Phi(x) = s(b^{(1)} + W^{(1)} x)`
    constitutes the hidden layer.  :math:`W^{(1)} \in R^{D \times D_h}` is
    the weight matrix connecting the input vector to the hidden layer.
    Each column :math:`W^{(1)}_{\cdot i}` represents the weights from the
    import input units to the i-th hidden unit. This estimator's :math:`s`
    is the Rectified linear unit output, or :math:`relu` function.

    Args:
        n_in (int): number of input nodes
        n_hidden (int or list(int)): if int this is the number of hidden layer
            nodes in a single-hidden-layer network. If list of int's this is
            a list of number of nodes for len(n_hidden) successive layers
        n_out (int): number of output nodes
        random (Optional(int or numpy.random.RandomState instance)):
            an integer seed or random state generator. Default: None, links to
            np.random
    Attributes:
        layers (list): List of all the estimator layers with layers[0] being
            the input layer, layer[1:-1] being the hidden layers and
            layers[-1] the output layer.
        X (theano variable): Symbolic input of first layer.
        y (theano variable): Symbolic output of last layer.
        params (list): Vector of all the estimator parameters, i.e. weights
            and biases of all the layers

        predict (theano expression): Return the most probable class
            (the probability function as described above).
        cost (theano expression): Negative log-likelihood from
            LogisticRegression.
    """

    def __init__(self, n_in, n_hidden, n_out, *args, **kwargs):
        super().__init__(n_in, n_hidden, n_out, LinearRegression,
                         relu, *args, **kwargs)


class MultiLayerPerceptron(MultiLayerBase, BaseEstimator):
    r""" Multilayer Perceptron.

    An MLR can be viewed as a logistic regression classifier where the input
    is first transformed using a learnt non-linear transformation
    :math:`\Phi`. This transformation projects the input data into a space
    where it becomes linearly separable. This intermediate layer is referred
    to as a **hidden layer**.

    Formally, a one-hidden-layer MLR is a function
    :math:`f: R^D \rightarrow R^L`, where :math:`D` is the size of input
    vector :math:`x` and :math:`L` is the size of the output vector
    :math:`f(x)`, such that, in matrix notation:

        .. math::

                f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

    with bias vectors :math:`b^{(1)}`, :math:`b^{(2)}`; weight matrices
    :math:`W^{(1)}`, :math:`W^{(2)}` and activation functions :math:`G` and
    :math:`s`. The vector :math:`h(x) = \Phi(x) = s(b^{(1)} + W^{(1)} x)`
    constitutes the hidden layer.  :math:`W^{(1)} \in R^{D \times D_h}` is
    the weight matrix connecting the input vector to the hidden layer.
    Each column :math:`W^{(1)}_{\cdot i}` represents the weights from the
    import input units to the i-th hidden unit. This estimator's :math:`s`
    is the :math:`tanh` function.

    Args:
        n_in (int): number of input nodes
        n_hidden (int or list(int)): if int this is the number of hidden layer
            nodes in a single-hidden-layer network. If list of int's this is
            a list of number of nodes for len(n_hidden) successive layers
        n_out (int): number of output nodes
        random (Optional(int or numpy.random.RandomState instance)):
            an integer seed or random state generator. Default: None, links to
            np.random
    Attributes:
        layers (list): List of all the estimator layers with layers[0] being
            the input layer, layer[1:-1] being the hidden layers and
            layers[-1] the output layer.
        X (theano variable): Symbolic input of first layer.
        y (theano variable): Symbolic output of last layer.
        params (list): Vector of all the estimator parameters, i.e. weights
            and biases of all the layers

        predict (theano expression): Return the most probable class
            (the probability function as described above).
        cost (theano expression): Negative log-likelihood from
            LogisticRegression.
    """
    def __init__(self, n_in, n_hidden, n_out, *args, **kwargs):
        super().__init__(n_in, n_hidden, n_out, LogisticRegression,
                         T.tanh, *args, **kwargs)


# TRANSFORMERS ###############################################################
class TiedAutoEncoder(RandomBase):
    def __init__(self, n_in, n_hidden, activation=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        activation = T.nnet.sigmoid if activation is None else activation

        self.layers = []

        if isinstance(n_hidden, float):
            n_hidden = int(n_in * n_hidden)
        self.layers.append(HiddenLayer(n_in, n_hidden,
                                       activation, self._rng))

        self.layers.append(HiddenLayer(n_hidden, n_in, activation, self._rng,
                                       X=self.layers[0].output))
        self.layers[1].W = self.layers[0].W.T
        self.params = [self.layers[0].W, self.layers[0].b, self.layers[1].b]

        self.X = self.layers[0].X
        self.transform = self.layers[0].output
        self.reconstruct = self.layers[1].output

        self.cost = T.mean(-T.sum(self.X * T.log(self.reconstruct) +
                                  (1 - self.X) * T.log(1 - self.reconstruct),
                                  axis=1))


class AutoEncoder(RandomBase):
    def __init__(self, n_in, n_hidden, activation=None, cost=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        activation = T.nnet.sigmoid if activation is None else activation

        self.layers = []
        if isinstance(n_hidden, int) or isinstance(n_hidden, float):
            if isinstance(n_hidden, float):
                n_hidden = int(n_in * n_hidden)
            if not isinstance(activation, list):
                activation = [activation, activation]
            n_prev = n_hidden
            self.layers.append(HiddenLayer(n_in, n_hidden,
                                           activation[0], self._rng))
        elif isinstance(n_hidden, list):
            if not isinstance(activation, list):
                temp = activation
                activation = [temp for i in range(len(n_hidden)+1)]
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

        self.layers.append(HiddenLayer(n_prev, n_in, activation[-1],
                                       X=self.layers[-1].output))

        self.params = [p for l in self.layers for p in l.params]

        self.X = self.layers[0].X
        self.transform = self.layers[-2].output
        self.reconstruct = self.layers[-1].output

        cost = 'squared' if cost is None else cost

        if isinstance(cost, str):
            if cost == 'squared':
                self.cost = T.sum(T.pow(T.abs_(self.X - self.reconstruct), 2))
            elif cost == 'cross_entropy':
                self.cost = T.mean(-T.sum(
                    self.X * T.log(self.reconstruct) + (1 - self.X)))
        else:
            self.cost = cost(self)
# pylint: enable=invalid-name
# pylint: enable=too-few-public-methods
