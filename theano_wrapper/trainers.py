import sys

import numpy as np
import theano
from theano import tensor as T

from theano_wrapper.common import RandomBase


class TrainerBase(RandomBase):
    """ Base class for trainers.
    Attributes:
        clf: (class) Classifier or Regressor
        X: (theano matrix) From clf
        y: (theano vector) From clf
        _verbose: (int) The verbosity factor. 0 = off
                       n = sys.stdout.write report every n-th period
        rng: (np.random.RandomState instance) RandomState generator
    Methods:
        _Xy_split_to_shared(X, y, lim): Splits the input samples X, y in
                                        training and validation set.
                                        Returns: [(X_train, y_train),
                                                  (X_test, y_test)]
    """
    def __init__(self, clf, verbose=None, **kwargs):
        """Arguements:
        clf: (class) Classifier or Regressor class
        verbose: (int) The verbosity factor. 0 = off
                       n = print report every n-th period
        """
        super().__init__(**kwargs)
        self.clf = clf
        self._verbose = verbose

        self.X = clf.X
        self.y = clf.y

    def _split_Xy_to_shared(self, X, y, lim=0.8):
        """ Split X and y into training and validation sets and store them
        on theano variables.
        Arguements:
            X: (arr(n_samples, n_features)): Input samples
            y: (arr(n_samples,)): Output predictions
            lim: (float) Limit for training set.
                         e.g. 0.6 = 60% train, 40% validation
                         default value: 0.8 (80% train, 20% validation)
        """
        per = self._rng.permutation(len(y))
        lim = int(len(y) * lim)
        train, test = per[:lim], per[lim:]

        X_train = theano.shared(np.asarray(X[train],
                                           dtype=theano.config.floatX),
                                borrow=True)
        X_test = theano.shared(np.asarray(X[test],
                                          dtype=theano.config.floatX),
                               borrow=True)
        y_train = theano.shared(np.asarray(y[train]), borrow=True)
        y_test = theano.shared(np.asarray(y[test]), borrow=True)

        return [(X_train, y_train), (X_test, y_test)]



class EpochTrainer(TrainerBase):
    """ Simple epoch-based trainer using Gradient Descent with patience.
    """
    def __init__(self, clf, alpha=0.01, max_iter=10000, patience=5000,
                 p_inc=2, imp_thresh=0.995, **kwargs):

        super().__init__(clf, **kwargs)

        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.p_inc = p_inc
        self.imp_thresh = imp_thresh
        self.cost = self.clf.cost

        self.gradients = [T.grad(self.cost, p) for p in self.clf.params]
        self.updates = [(p, p - self.alpha * g)
                        for p, g in zip(self.clf.params, self.gradients)]

    def fit(self, X, y):
        train_set, val_set = self._split_Xy_to_shared(X, y)
        self._init_models(train_set, val_set)
        val_freq = self.patience/3
        patience = self.patience
        best_val_loss = np.inf

        for i in range(self.max_iter):
            epoch_loss = float(self.train_model())
            if self._verbose:
                if (i+1) % self._verbose == 0:
                    sys.stdout.write(
                        "Epoch {:5d}, this loss: {:8.3f}, "
                        "best_val_loss:{:7.3f} "
                        "{:6d} more samples to go\r".format(
                            (i+1), epoch_loss, best_val_loss,
                            int(patience-i)))
            if i % val_freq == 0:
                val_loss = float(self.val_model())
                if val_loss < best_val_loss:
                    if val_loss < best_val_loss * self.imp_thresh:
                        patience = max(patience, i * self.p_inc)
                    best_val_loss = val_loss
            if patience <= i:
                if self._verbose:
                    print("\nClassifier converged")
                break
        else:
            if self._verbose:
                print("\nMaximum iterations reached.")

    def _init_models(self, train_set, val_set):
        self.train_model = theano.function(
            inputs=[], outputs=self.cost, updates=self.updates,
            givens={self.X: train_set[0], self.y: train_set[1]})

        self.val_model = theano.function(
            inputs=[], outputs=self.cost,
            givens={self.X: val_set[0], self.y: val_set[1]})

        self.predict_model = theano.function(inputs=[self.X],
                                             outputs=self.clf.predict)

    def predict(self, X):
        if hasattr(self, 'predict_model'):
            return self.predict_model(X)
        else:
            # handle the exception
            raise AttributeError("Classifier hasn't been fitted yet")


