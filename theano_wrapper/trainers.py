""" Various trainers for machine learning applications.
Classes:
    EpochTrainer: A simple epoch-based trainer
"""
import sys

import numpy as np
import theano
from theano import tensor as T

from theano_wrapper.common import RandomBase


# pylint: disable=invalid-name
# Names like X,y, X_train, y_train etc. are common in machine learning
# tasks. For better readability and comprehension, disable pylint on
# invalid names.

# pylint: disable=too-few-public-methods
# This is a base class to be inherited from
class TrainerBase(RandomBase):
    """ Base class for trainers.
    Attributes:
        clf (class): Classifier or Regressor
        X (theano matrix): From clf
        y (theano vector): From clf
        _verbose (int): The verbosity factor. 0 = off
                       n = sys.stdout.write report every n-th period
        rng (np.random.RandomState instance): RandomState generator
    Methods:
        _Xy_split_to_shared(X, y, lim): Splits the input samples X, y in
                                        training and validation set.
                                        Returns: [(X_train, y_train),
                                                  (X_test, y_test)]
    """
    def __init__(self, clf, verbose=None, random=None):
        """Arguements:
        clf: (class) Classifier or Regressor class
        verbose: (int) The verbosity factor. 0 = off
                       n = print report every n-th period
        """
        super().__init__(random)
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


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# It is inevitable to have that many arguements and parameters for a training
# class.
class GradientDescentBase(TrainerBase):
    """ Base for gradient descent trainers with patience
    Args:
        clf The estimator to train
        alpha (float): learning rate
        max_iter (int): max_iterations to go through
        patience (int): look at least that many samples
        p_inc (float): how many more samples to fit after each improvement
        imp_thresh (float): the limit of what to consider improvement
        random (int or random state generator): from TrainerBase
        verbose (int): from TrainerBase
    Attributes:
        gradients (theano symbolic function): The gradient for each parameter
        updates (theano symbolic function): Compute update values
    Methods:
        fit(X, y): X: (arr(n_samples, n_features))
                   y: (arr(n_samples, n_features))
                   Train estimator using input samples
                   This implementation will automatically split the input
                   into an 80% training and an 20% validation set
        predict(X): Return estimator prediction for input X
    """
    def __init__(self, clf, alpha=0.01, max_iter=10000, patience=5000,
                 p_inc=2.0, imp_thresh=0.995, *args, **kwargs):

        super().__init__(clf, *args, **kwargs)

        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.p_inc = p_inc
        self.imp_thresh = imp_thresh
        self.cost = self.clf.cost

        self.gradients = [T.grad(self.cost, p) for p in self.clf.params]
        self.updates = [(p, p - self.alpha * g)
                        for p, g in zip(self.clf.params, self.gradients)]

        self.train_model = None
        self.val_model = None
        self.predict_model = None

    def fit(self, X, y):
        """ Children class should implement a fit function
        input: X: arr(n_samples, n_features), y: arr(n_samples,)
        output: None, train the model
        """
        pass

    def _init_models(self, train_set, val_set):
        pass

    def predict(self, X):
        """ Predict y given X """
        if hasattr(self, 'predict_model'):
            return self.predict_model(X)   # pylint: disable=not-callable
        else:
            # handle the exception
            raise AttributeError("Classifier hasn't been fitted yet")
# pylint: enable=too-few-public-methods
# pylint: enable=too-many-arguments


class EpochTrainer(GradientDescentBase):
    """ Simple epoch-based trainer using Gradient Descent with patience.
    The idea is that we train for at least n  (`patience`) epochs and then if
    the score keeps getting better (biased by `imp_thresh`) we elongate the
    training session by a factor of `p_inc`.

    Args:
        clf: the estimator to train
        alpha (float): learning rate
        max_iter (int): max_iterations to go through
        patience (int): look at least that many samples
        p_inc (float): how many more samples to fit after each improvement
        imp_thresh (float): the limit of what to consider improvement
        random (int or random state generator): a random state for predictable
            results
        verbose (int): verbosity factor. None = off, n = every n periods

    Attributes:
        gradients (theano symbolic function): The gradient for each parameter.
        updates (theano symbolic function): Compute update values.

    Methods:
        fit(X, y): Train estimator using input samples. This implementation
            will automatically split the input into an 80% training and an
            20% validation set
        predict(X): Return estimator prediction for input X
    """
    def __init__(self, clf, *args, **kwargs):

        super().__init__(clf, *args, **kwargs)

    def fit(self, X, y):
        """ Split the input into train and validation set and
        run gradient-descent to find optimal model parameters
        Args:
            X (numpy array): Input matrix of shape: (n_samples, n_features)
            y (numpy array): Target values, shape: (n_samples,)
        """
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


class SGDTrainer(GradientDescentBase):
    """ Simple epoch-based trainer using Gradient Descent with patience.
    The idea is that we train for at least n  (=`patience`) epochs and then if
    the score keeps getting better (biased by `imp_thresh`) we elongate the
    training session by a factor of `p_inc`.

    Args:
        clf the estimator to train
        batch_size (int or None): how many samples to consider for each
            training batch. if None, it is set to int(n_samples/100)
        alpha (float): learning rate
        max_iter (int): max_iterations to go through
        patience (int): look at least that many samples
        p_inc (float): how many more samples to fit after each improvement
        imp_thresh (float): the limit of what to consider improvement
        random (int or random state generator): a random state for predictable
            resi;ts
        verbose (int): verbosity factor. None = off, n = every n periods

    Attributes:
        gradients: (theano symbolic function) The gradient for each parameter
        updates: (theano symbolic function) Compute update values

    Methods:
        fit(X, y): Train estimator using input samples. This implementation
            will automatically split the input into an 80% training and an
            20% validation set
        predict(X): Return estimator prediction for input X
    """
    def __init__(self, clf, batch_size=None, *args, **kwargs):

        super().__init__(clf, *args, **kwargs)
        self.batch_size = batch_size
        self.index = T.lscalar()    # minibatch index

    def fit(self, X, y):
        """ Split the input into train and validation set and
        run gradient-descent to find optimal model parameters
        """
        # pylint: disable=too-many-branches
        # maybe I should get rid of the double patience <= iteration
        # statement at l.284
        if not self.batch_size:
            self.batch_size = int(X.shape[0]/100)

        train_set, val_set = self._split_Xy_to_shared(X, y)
        n_train_batches, n_val_batches = self._get_minibatches(train_set,
                                                               val_set)
        self._init_models(train_set, val_set)
        val_freq = min(n_train_batches, self.patience/3)
        patience = self.patience
        best_val_loss = np.inf

        for i in range(self.max_iter):
            for batch in range(n_train_batches):
                batch_loss = float(self.train_model(batch))
                iteration = i * n_train_batches + batch
                if self._verbose:
                    if (batch+1) % self._verbose == 0:
                        sys.stdout.write(
                            "Epoch {:4d}, minibatch {:4d}/{:4d}, "
                            "test loss: {:8.3f}, best_val_loss:{:7.3f} "
                            "{:6d} more samples to go\r".format(
                                (i+1), (batch+1), n_train_batches,
                                batch_loss, best_val_loss,
                                int(patience-iteration)))

                if (iteration+1) % val_freq == 0:
                    val_loss = np.mean([self.val_model(b)
                                        for b in range(n_val_batches)])
                    if val_loss < best_val_loss:
                        if val_loss < best_val_loss * self.imp_thresh:
                            patience = max(patience, iteration * self.p_inc)
                        best_val_loss = val_loss

                if patience <= iteration:
                    break
            if patience <= iteration:
                if self._verbose:
                    print("\nClassifier converged")
                break
        else:
            if self._verbose:
                print("\nMaximum iterations reached.")
        # pylint: enable=too-many-branches

    def _get_minibatches(self, set1, set2):
        n1 = int(set1[0].get_value(borrow=True).shape[0] / self.batch_size)
        n2 = int(set2[0].get_value(borrow=True).shape[0] / self.batch_size)
        return n1, n2

    def _init_models(self, train_set, val_set):
        self.train_model = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            updates=self.updates,
            givens={
                self.X: train_set[0][self.index*self.batch_size:
                                     (self.index+1)*self.batch_size],
                self.y: train_set[1][self.index*self.batch_size:
                                     (self.index+1)*self.batch_size]})
        self.val_model = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            givens={
                self.X: val_set[0][self.index*self.batch_size:
                                   (self.index+1)*self.batch_size],
                self.y: val_set[1][self.index*self.batch_size:
                                   (self.index+1)*self.batch_size]})
        self.predict_model = theano.function(inputs=[self.X],
                                             outputs=self.clf.predict)
# pylint: enable=invalid-name
