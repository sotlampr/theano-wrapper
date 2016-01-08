import unittest

import numpy as np

from theano_wrapper.demo import demo


class TestDemo(unittest.TestCase):
    def test_demo_regression_1(self):
        try:
            demo('r1', True)
        except Exception as e:
            self.fail("Demo raised exception %s" % str(e))

    def test_demo_regression_2(self):
        try:
            demo('r2', True)
        except Exception as e:
            self.fail("Demo raised exception %s" % str(e))

    def test_demo_classification_1(self):
        try:
            demo('c1', True)
        except Exception as e:
            self.fail("Demo raised exception %s" % str(e))

    def test_demo_classification_2(self):
        try:
            demo('c2', True)
        except Exception as e:
            self.fail("Demo raised exception %s" % str(e))

    def test_demo_classification_3(self):
        try:
            demo('c3', True)
        except Exception as e:
            self.fail("Demo raised exception %s" % str(e))

    def test_demo_classification_4(self):
        try:
            demo('c4', True)
        except Exception as e:
            self.fail("Demo raised exception %s" % str(e))
