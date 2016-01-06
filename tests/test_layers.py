import unittest
import sys

import numpy as np
import theano

from theano_wrapper.layers import BaseLayer


class TestLayer(unittest.TestCase):
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
