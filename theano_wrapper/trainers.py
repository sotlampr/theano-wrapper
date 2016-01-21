""" Various trainers for machine learning applications.
Classes:
    EpochTrainer: A simple epoch-based trainer
"""
import sys

import numpy as np
import theano
from theano import tensor as T

from theano_wrapper.common import RandomBase


# REGULARIZERS ###############################################################
def l1_l2_reg(clf, l1_reg=.0, l2_reg=.0001):
    r""" L1 and L2 squared regularization.

    L1 and L2 regularization involve adding an extra term to the loss function,
    which penalizes certain parameter configurations. For a loss function
    :math:`\ell(\theta, \cal{D})` of the prediction function f parameterized
    by :math:`\theta` on data set :math:`\cal{D}`, the regularized loss
    will be:

    .. math::

        E(\theta, \mathcal{D}) =  \ell(\theta, \mathcal{D}) +
                                  \lambda R(\theta)\\
    or, in our case

    .. math::

        E(\theta, \mathcal{D}) =  NLL(\theta, \mathcal{D}) +
                                  \lambda||\theta||_p^p

    where

    .. math::

        ||\theta||_p =
        \left(\sum_{j=0}^{|\theta|}{|\theta_j|^p}\right)^{\frac{1}{p}}

    :math:`\theta` is a set of all parameters for a given model,
    :math:`\lambda` the hyper-parameter which controls the relative
    importance of the regularization parameter and :math:`R` the
    regularization function. Commonly used values for :math:`p`
    are 1 and 2, hence the L1/L2 nomenclature. If :math:`p=2`, then the
    regularizer is also called "weight decay".

    In this model both L1 and L2 regularization is supported.

    Args:
        clf: an estimator
        l1_reg (float): The l1 regularization parameter. Defaults to .0
        l2_reg (float): The l2 regularization parameter. Defaults to .0001
    Returns:
        cost (theano expression): Symbolic expression that calculates the
            regularized cost.
    Example::

        clf = SomeClassifier(*args)
        reg = l1_l2_reg(clf, 0.0001, 0.001)
        trn = SomeTrainer(clf, reg=reg)
        [...]

    """
    weights = (p for p in clf.params if p.name == 'W')

    L1 = T.sum([T.sum(abs(w)) for w in weights])
    L2_sqr = T.sum([T.sum(T.pow(w, 2)) for w in weights])

    return clf.cost + l1_reg * L1 + l2_reg * L2_sqr


# BASE CLASSES ###############################################################
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
    def __init__(self, clf, reg=None, preprocessor=None,
                 verbose=None, random=None):
        """Arguements:
        clf: (class) Classifier or Regressor class
        verbose: (int) The verbosity factor. 0 = off
                       n = print report every n-th period
        """
        super().__init__(random)
        self.clf = clf
        self._verbose = verbose

        self.cost = reg if reg else self.clf.cost

        self.X = clf.X
        try:
            self.y = clf.y
            self.__clf_type = 'estimator'
        except AttributeError:
            self.y = None
            self.__clf_type = 'transformer'

        self.pre = preprocessor if preprocessor else None

    def fit(self, X, y=None):
        if self.pre:
            X = self.pre.fit_transform(X)
        if y is None:
            self._fit_transformer(X)
        else:
            self._fit_estimator(X, y)

    def _fit(self, train_set, val_set):
        """ Children class should implement a _fit function for fitting
        both tranformers and estimators
        """

    def _fit_transformer(self, X):
        """ Fit function for transfomers
        input: X: arr(n_samples, n_features))
        output: None, train the model
        """
        self._fit(*self._split_X_to_shared(X))

    def _fit_estimator(self, X, y):
        """ Fit function for estimators
        input: X: arr(n_samples, n_features)), y: arr(n_samples)
        output: None, train the model
        """
        self._fit(*self._split_Xy_to_shared(X, y))

    def _init_models(self, train_set, val_set):
        pass

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

    def _split_X_to_shared(self, X, lim=0.8):
        """ Split X into training and validation sets and store them
        on theano variables.
        Arguements:
            X: (arr(n_samples, n_features)): Input samples
            lim: (float) Limit for training set.
                         e.g. 0.6 = 60% train, 40% validation
                         default value: 0.8 (80% train, 20% validation)
        """
        per = self._rng.permutation(len(X))
        lim = int(len(X) * lim)
        train, test = per[:lim], per[lim:]

        X_train = theano.shared(np.asarray(X[train],
                                           dtype=theano.config.floatX),
                                borrow=True)
        X_test = theano.shared(np.asarray(X[test],
                                          dtype=theano.config.floatX),
                               borrow=True)

        return [(X_train,), (X_test,)]

    def predict(self, X):
        """ Predict y given X """
        if hasattr(self, 'predict_model'):
            if self.pre:
                X = self.pre.transform(X)
            return self.predict_model(X)
        else:
            # handle the exception
            raise AttributeError("Estimator hasn't been fitted yet")

    def transform(self, X):
        if self.__clf_type == 'estimator':
            # handle the exception
            raise AttributeError("Given clf is an estimator")
        else:
            if hasattr(self, 'transform_model'):
                if self.pre:
                    X = self.pre.transform(X)
                return self.transform_model(X)
            else:
                # handle the exception
                raise AttributeError("Transformer hasn't been fitted yet")


# TRAINERS ###################################################################
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
    def __init__(self, clf, alpha=0.01, max_iter=1000, patience=500,
                 p_inc=2.0, imp_thresh=0.995, *args, **kwargs):

        super().__init__(clf, *args, **kwargs)

        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.p_inc = p_inc
        self.imp_thresh = imp_thresh

        self.gradients = [T.grad(self.cost, p) for p in self.clf.params]
        self.updates = [(p, p - self.alpha * g)
                        for p, g in zip(self.clf.params, self.gradients)]

        self.train_model = None
        self.val_model = None
        self.predict_model = None
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

    def _fit(self, train_set, val_set):
        """ Split the input into train and validation set and
        run gradient-descent to find optimal model parameters
        Args:
            X (numpy array): Input matrix of shape: (n_samples, n_features)
            y (numpy array): Target values, shape: (n_samples,)
        """
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
        if self.y is not None:
            givens_train = {self.X: train_set[0], self.y: train_set[1]}
            givens_val = {self.X: val_set[0], self.y: val_set[1]}
        else:
            givens_train = {self.X: train_set[0]}
            givens_val = {self.X: val_set[0]}
            self.transform_model = theano.function(inputs=[self.X],
                                                   outputs=self.clf.encode)

        self.train_model = theano.function(
            inputs=[], outputs=self.cost, updates=self.updates,
            givens=givens_train)

        self.val_model = theano.function(
            inputs=[], outputs=self.cost,
            givens=givens_val)

        self.predict_model = theano.function(inputs=[self.X],
                                             outputs=self.clf.output)


class SGDTrainer(GradientDescentBase):
    """ Stohastic Gradient descent trainer with patience.
    This classifier works in a similar way to EpochTrainer, but instead of
    fitting all the samples it splits them to minibatches and go through
    a subset of all the samples at a fit period. This allows for speed
    improvements with large datasets and off-line training, i.e. training
    without all the samples available at once.

    Args:
        clf: the estimator to train
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

    def _fit(self, train_set, val_set):
        """ Split the input into train and validation set and
        run gradient-descent to find optimal model parameters
        """
        # pylint: disable=too-many-branches
        # maybe I should get rid of the double patience <= iteration
        # statement at l.284
        if not self.batch_size:
            self.batch_size = int(train_set[0].get_value().shape[0]/100)

        n_train_batches, n_val_batches = self._get_minibatches(train_set,
                                                               val_set)
        self._init_models(train_set, val_set)
        val_freq = int(min(n_train_batches*2, self.patience/2))
        patience = self.patience
        best_val_loss = np.inf

        for i in range(self.max_iter):
            for batch in range(n_train_batches):
                batch_loss = float(self.train_model(batch))
                iteration = (i * n_train_batches) + batch
                if self._verbose:
                    if (batch+1) % self._verbose == 0:
                        sys.stdout.write(
                            "Epoch {:4d}, minibatch {:4d}/{:4d}, "
                            "test loss: {:8.3f}, best_val_loss:{:7.3f} "
                            "{:6d} more batches to go\r".format(
                                (i+1), (batch+1), n_train_batches,
                                batch_loss, best_val_loss,
                                int(patience-iteration)))

                if (iteration+1) % val_freq == 0:
                    val_loss = np.mean([self.val_model(b)
                                        for b in range(n_val_batches)])
                    if val_loss < best_val_loss:
                        if val_loss < best_val_loss * self.imp_thresh:
                            patience = int(max(patience,
                                               iteration * self.p_inc))
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
        if self.y is not None:
            givens_train = {
                self.X: train_set[0][self.index*self.batch_size:
                                     (self.index+1)*self.batch_size],
                self.y: train_set[1][self.index*self.batch_size:
                                     (self.index+1)*self.batch_size]}
            givens_val = {
                self.X: val_set[0][self.index*self.batch_size:
                                   (self.index+1)*self.batch_size],
                self.y: val_set[1][self.index*self.batch_size:
                                   (self.index+1)*self.batch_size]}
        else:
            givens_train = {
                self.X: train_set[0][self.index*self.batch_size:
                                     (self.index+1)*self.batch_size]}
            givens_val = {
                self.X: val_set[0][self.index*self.batch_size:
                                   (self.index+1)*self.batch_size]}
            self.transform_model = theano.function(inputs=[self.X],
                                                   outputs=self.clf.encode)

        self.train_model = theano.function(
            inputs=[self.index], outputs=self.cost,
            updates=self.updates, givens=givens_train)

        self.val_model = theano.function(
            inputs=[self.index], outputs=self.cost,
            givens=givens_val)

        self.predict_model = theano.function(inputs=[self.X],
                                             outputs=self.clf.output)

# pylint: enable=invalid-name
