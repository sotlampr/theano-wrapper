import sys
from io import StringIO
import unittest

import numpy as np

from theano_wrapper.trainers import TrainerBase, EpochTrainer
from tests.helpers import SimpleClf


class TestBase(unittest.TestCase):
    """ Tests for trainers.py module, which includes various trainer classes
    for use with theano-wrapper
    """
    def test_trainer_base_initialization(self):
        try:
            trb = TrainerBase(SimpleClf())
        except Exception as e:
            self.fail("Class initialization failed: %s" % str(e))

    def test_trainer_base_intergrates_clf(self):
        clf = SimpleClf()
        trb = TrainerBase(clf)
        self.assertIs(trb.clf, clf)
        self.assertIs(trb.X, clf.X)
        self.assertIs(trb.y, clf.y)

    def test_trainer_base_verbosity_factor(self):
        trb = TrainerBase(SimpleClf(), 42)
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


class BaseTrainerTest:
    def test_trainer_initialization(self):
        try:
            etrain = self.trainer(SimpleClf())
        except Exception as e:
           self.fail("Class initialization failed: %s" % str(e))

    def test_trainer_fit_default(self):
        X = np.random.random_sample((500, 100))
        y = np.random.randint(0, 9, (500,)).astype(np.int32)
        etrain = self.trainer(SimpleClf())
        try:
            etrain.fit(X, y)
        except Exception as e:
           self.fail("Training failed: %s" % str(e))


    def test_trainer_fit_converge(self):
        X = np.random.random_sample((500, 100))
        y = np.random.randint(0, 9, (500,)).astype(np.int32)
        etrain = self.trainer(SimpleClf(), patience=100, max_iter=1000,
                              verbose=.1)
        saved_stdout = sys.stdout
        out = StringIO()
        sys.stdout = out
        etrain.fit(X, y)
        output = out.getvalue().strip()
        sys.stdout = saved_stdout
        self.assertIn('converge', output.lower())

    def test_trainer_fit_reach_max(self):
        X = np.random.random_sample((500, 100))
        y = np.random.randint(0, 9, (500,)).astype(np.int32)
        etrain = self.trainer(SimpleClf(), patience=1000, max_iter=100,
                              verbose=.1)
        saved_stdout = sys.stdout
        out = StringIO()
        sys.stdout = out
        etrain.fit(X, y)
        output = out.getvalue().strip()
        sys.stdout = saved_stdout
        self.assertIn('maximum', output.lower())

    def test_trainer_predict(self):
        X = np.random.random_sample((500, 100))
        y = np.random.randint(0, 9, (500,)).astype(np.int32)
        etrain = self.trainer(SimpleClf(), max_iter=100)
        etrain.fit(X, y)
        X_test = np.random.random_sample((100, 100))
        y_pred = etrain.predict(X_test)
        self.assertEquals(y_pred.shape, (100,))
        y_values = np.unique(y)
        for targ in np.unique(y_pred):
            self.assertIn(targ, y_values,
                          msg="Output contains value non-existent in "
                              "training set")



class TestEpochTrainer(unittest.TestCase, BaseTrainerTest):
    trainer = EpochTrainer
