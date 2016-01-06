from theano_wrapper.common import RandomBase

class TrainerBase(RandomBase):
    """ Base class for trainers.
    Attributes:
        clf: (class) Classifier or Regressor
        X: (theano matrix) From clf
        y: (theano vector) From clf
        _verbose: (int) The verbosity factor. 0 = off
                       n = print report every n-th period
        rng: (np.random.RandomState instance) RandomState generator
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
