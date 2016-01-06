import unittest

import numpy as np

from theano_wrapper.trainers import TrainerBase
from tests.helpers import DummyClf


class TestTrainers(unittest.TestCase):
    """ Tests for trainers.py module, which includes various trainer classes
    for use with theano-wrapper
    """
    def test_trainer_base_initialization(self):
        try:
            trb = TrainerBase(DummyClf())
        except Exception as e:
            self.fail("Class initialization failed: %s" % str(e))

    def test_trainer_base_intergrates_clf(self):
        clf = DummyClf()
        trb = TrainerBase(clf)
        self.assertIs(trb.clf, clf)
        self.assertIs(trb.X, clf.X)
        self.assertIs(trb.y, clf.y)

    def test_trainer_base_verbosity_factor(self):
        trb = TrainerBase(DummyClf(), 42)
        self.assertEqual(trb._verbose, 42)

    def test_trainer_base_random_state_gen_no_args(self):
        trb = TrainerBase(DummyClf())
        self.assertTrue(hasattr(trb, '_rng'))
        self.assertEqual(trb._rng, np.random)

    def test_trainer_base_random_state_gen_random(self):
        trb = TrainerBase(DummyClf(), random=42)
        self.assertIsInstance(trb._rng, np.random.RandomState)
