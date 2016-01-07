import unittest
import sys

import numpy as np
import theano

from tests.helpers import SimpleTrainer
from theano_wrapper.layers import BaseLayer, LinearRegression


class TestBaseLayer(unittest.TestCase):
    """ Tests for layer.py module, which includes various types of layers
    for theano-wrapper
    """

    def test_base_layer_has_params(self):
        base = BaseLayer(100, 10)
        self.assertTrue(hasattr(base, 'params'),
                        msg="Class has no attribute 'parameters'")

    def test_base_layer_params_not_empty(self):
        base = BaseLayer(100, 10)
        self.assertTrue(base.params, msg="Class 'parameters' are empty")

    def test_base_layer_no_args(self):
        # Test if BaseLayer initializes as expected when given no
        # extra arguements
        try:
            base = BaseLayer(100, 10)
        except Exception as e:
            self.fail("Class initialization failed: %s" % str(e))

    def test_base_layer_params_are_theano_shared_variables(self):
        base = BaseLayer(100, 10)
        for p in base.params:
            self.assertIsInstance(p, theano.compile.SharedVariable)

    def test_base_layer_has_input(self):
        base = BaseLayer(100, 10)
        self.assertTrue(hasattr(base, 'X'))

    def test_base_layer_input_is_theano_variable(self):
        base = BaseLayer(100, 10)
        self.assertIsInstance(base.X, theano.tensor.TensorVariable)

    def test_base_layer_weights_shape(self):
        base = BaseLayer(100, 10)
        self.assertEqual(base.W.get_value().shape, (100, 10))

    def test_base_layer_bias_shape(self):
        base = BaseLayer(100, 10)
        self.assertEqual(base.b.get_value().shape, (10,))

    def test_base_layer_weights_shape_single_output(self):
        base = BaseLayer(100, 1)
        self.assertEqual(base.W.get_value().shape, (100,))

    def test_base_layer_bias_shape_single_output(self):
        base = BaseLayer(100, 1)
        self.assertEqual(base.b.get_value().shape, ())

    def test_base_layer_no_output(self):
        base = BaseLayer(100, 10)
        self.assertFalse(hasattr(base, 'y'))

    def test_base_layer_int_output(self):
        base = BaseLayer(100, 10, y='int')
        self.assertTrue(hasattr(base, 'y'))
        self.assertTrue(hasattr(base.y, 'dtype'))
        self.assertEqual(base.y.dtype, 'int32')

    def test_base_layer_float_output(self):
        base = BaseLayer(100, 10, y='float')
        self.assertTrue(hasattr(base, 'y'))
        self.assertTrue(hasattr(base.y, 'dtype'))
        self.assertEqual(base.y.dtype, 'float32')


class EstimatorTest:
    X = np.random.standard_normal((500, 100)).astype(np.float32)
    def test_estimator_has_params(self):
        clf = self.estimator(*self.l_shape)
        self.assertTrue(hasattr(clf, 'params'))
        self.assertIsNotNone(clf.params)

    def test_estimator_fit(self):
        trn = SimpleTrainer(self.estimator(*self.l_shape))
        try:
            trn.fit(self.X, self.y)
        except Exception as e:
            self.fail("Training failed: %s" % str(e))


class ClassificationTest(EstimatorTest):
    l_shape = (100, 10)
    y = np.random.randint(0, 9, (500,)).astype(np.int32)


class RegressionTest(EstimatorTest):
    l_shape = (100, 1)
    y= np.random.random((500,)).astype(np.float32)


class TestLinearRegression(unittest.TestCase, RegressionTest):
    estimator = LinearRegression
