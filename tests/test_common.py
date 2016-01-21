import unittest

import numpy as np
import theano

from theano_wrapper.common import RandomBase


class TestCommon(unittest.TestCase):
    """ Tests for common.py module, which includes common classes and methods
    for theano-wrapper
    """
    @staticmethod
    def get_random_base_dummy_class():
        # Initialize a dummy class that inherits from RandomBase
        class Dummy(RandomBase):
            def __init__(self, *args, **kwargs):
                super(Dummy, self).__init__(*args, **kwargs)

        return Dummy

    def test_random_base_init(self):
        Dummy = self.get_random_base_dummy_class()
        try:
            Dummy(1)
        except Exception as e:
            print("Class initialization failed: %s" % str(e))

    def test_random_base_no_arguements(self):
        # If we pass no arguements to RandomBase it should link to the
        # np.random namespace
        Dummy = self.get_random_base_dummy_class()
        tester = Dummy()
        self.assertEqual(tester._rng, np.random)

    def test_random_base_integer_seed(self):
        Dummy = self.get_random_base_dummy_class()
        seed = np.random.randint(0, 10000)
        tester = Dummy(seed)
        self.assertEqual(tester._rng.__getstate__()[1][0], seed)

    def test_random_base_generator(self):
        # Test if passing a random state generator works
        Dummy = self.get_random_base_dummy_class()
        rng = np.random.RandomState(42)
        tester = Dummy(rng)
        self.assertEqual(tester._rng, rng)

    def test_theano_rng(self):
        Dummy = self.get_random_base_dummy_class()
        tester = Dummy()
        self.assertIsInstance(tester._srng,
                              theano.sandbox.rng_mrg.MRG_RandomStreams)
