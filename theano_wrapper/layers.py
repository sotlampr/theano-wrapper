""" Various layers for machine learning applications """
import numpy as np
import theano
from theano import tensor as T


class BaseLayer:
    def __init__(self, n_in, n_out, *args, **kwargs):
        self.X = T.matrix('X')
        self.__init_weights_bias(n_in, n_out)


    def __init_weights_bias(self, n_in, n_out):
        # Weights and bias initialization
        if n_out == 1:
            _weights = np.zeros((n_in), dtype=theano.config.floatX)
            _bias = 0.
        else:
            _weights = np.zeros((n_in, n_out), dtype=theano.config.floatX)
            _bias = np.zeros(n_out,)

        self.W = theano.shared(_weights, name='W')
        self.b = theano.shared(_bias, name='b')
        self.params = [self.W, self.b]
