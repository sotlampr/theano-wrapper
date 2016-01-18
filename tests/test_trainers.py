import sys
from io import StringIO
import unittest

import numpy as np
import theano

from theano_wrapper.trainers import (TrainerBase, l1_l2_reg,
                                     EpochTrainer, SGDTrainer)
from tests.helpers import SimpleClf, SimpleTransformer


class TestBase(unittest.TestCase):
    """ Tests for trainers.py module, which includes various trainer classes
    for use with theano-wrapper
    """
    def test_trainer_base_initialization(self):
        try:
            TrainerBase(SimpleClf())
        except Exception as e:
            self.fail("Class initialization failed: %s" % str(e))

    def test_trainer_base_intergrates_clf(self):
        clf = SimpleClf()
        trb = TrainerBase(clf)
        self.assertIs(trb.clf, clf)
        self.assertIs(trb.X, clf.X)
        self.assertIs(trb.y, clf.y)

    def test_trainer_base_verbosity_factor(self):
        trb = TrainerBase(SimpleClf(), verbose=42)
        self.assertEqual(trb._verbose, 42)

    def test_trainer_base_random_state_gen_no_args(self):
        trb = TrainerBase(SimpleClf())
        self.assertTrue(hasattr(trb, '_rng'))
        self.assertEqual(trb._rng, np.random)

    def test_trainer_base_random_state_gen_random(self):
        trb = TrainerBase(SimpleClf(), random=42)
        self.assertIsInstance(trb._rng, np.random.RandomState)

    def test_trainer_base_split_to_shared(self):
        trb = TrainerBase(SimpleClf())
        X = np.random.random_sample((1000, 100))
        y = np.random.randint(0, 3, (1000,))
        train_set, test_set = trb._split_Xy_to_shared(X, y, 0.6)
        self.assertEqual(train_set[0].get_value().shape, (600, 100))
        self.assertEqual(train_set[1].get_value().shape, (600, ))
        self.assertEqual(test_set[0].get_value().shape, (400, 100))
        self.assertEqual(test_set[1].get_value().shape, (400, ))


class BaseTrainerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if cls is BaseTrainerTest:
            raise unittest.SkipTest("Skip BaseTest tests, it's a base class")
        super().setUpClass()
        cls.X = np.random.sample((500, 100)).astype(
            theano.config.floatX)
        cls.y = np.random.randint(0, 9, (500,)).astype(np.int32)
        cls.X_test = np.random.random_sample((100, 100)).astype(
            theano.config.floatX)

    def test_trainer_initialization(self):
        try:
            self.trainer(SimpleClf())
        except Exception as e:
            self.fail("Class initialization failed: %s" % str(e))

    def test_trainer_fit_default(self):
        etrain = self.trainer(SimpleClf())
        try:
            etrain.fit(self.X, self.y)
        except Exception as e:
            self.fail("Training failed: %s" % str(e))

    def test_trainer_fit_converge(self):
        etrain = self.trainer(SimpleClf(), patience=100, max_iter=1000,
                              verbose=.1)
        saved_stdout = sys.stdout
        out = StringIO()
        sys.stdout = out
        etrain.fit(self.X, self.y)
        output = out.getvalue().strip()
        sys.stdout = saved_stdout
        self.assertIn('converge', output.lower())

    def test_trainer_fit_reach_max(self):
        etrain = self.trainer(SimpleClf(), patience=1000, max_iter=10,
                              verbose=.1)
        saved_stdout = sys.stdout
        out = StringIO()
        sys.stdout = out
        etrain.fit(self.X, self.y)
        output = out.getvalue().strip()
        sys.stdout = saved_stdout
        self.assertIn('maximum', output.lower())

    def test_trainer_predict(self):
        etrain = self.trainer(SimpleClf(), patience=1000, max_iter=10)
        etrain.fit(self.X, self.y)
        y_pred = etrain.predict(self.X_test)
        self.assertEquals(y_pred.shape, (100,))
        y_values = np.unique(self.y)
        for targ in np.unique(y_pred):
            self.assertIn(targ, y_values,
                          msg="Output contains value non-existent in "
                              "training set")

    def test_trainer_with_transformer(self):
        etrain = self.trainer(SimpleTransformer(), max_iter=3)
        try:
            etrain.fit(self.X)
        except Exception as e:
            self.fail("Training failed: %s" % str(e))

    def test_trainer_transform(self):
        etrain = self.trainer(SimpleTransformer(), max_iter=3)
        etrain.fit(self.X)
        try:
            etrain.transform(self.X[:10])
        except Exception as e:
            self.fail("Failed: %s" % str(e))

    def test_train_with_regularization(self):
        clf = SimpleClf()
        reg = l1_l2_reg(clf, 0.01, 0.01)
        try:
            etrain = self.trainer(clf, reg=reg, max_iter=10)
            etrain.fit(self.X, self.y)
        except Exception as e:
            self.fail("Trainer failed: %s" % str(e))


class TestEpochTrainer(BaseTrainerTest):
    trainer = EpochTrainer


class TestSGDTrainer(BaseTrainerTest):
    trainer = SGDTrainer
