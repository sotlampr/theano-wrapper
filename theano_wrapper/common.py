""" Common classes and methods for theano-wrapper
Classes
    RandomBase: A base class providing a random state generator
"""

import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


# pylint: disable=too-few-public-methods
# This is a base class to be inherited from
class RandomBase:
    """ A Base class providing a random state generator.
    Arguments
        random: (int or np.random.RandomState instance)
                An integer seed or a random state generator. If None is
                passed, instead of a random state generator the class
                provides the np.random namespace
    Attributes
        _rng: (np.random.RandomState instance)
             A random state generator
    Usage
        >>> class NeedsRandom(RandomBase):
                def __init__(self. **kwargs):
                   super(NeedsRandom, self).__init__(**kwargs)

        >>> randomized = NeedsRandom(random=42)
        >>> type(randomized._rng)
        mt.rand.RandomState
    """
    def __init__(self, random=None):
        if random is None or random is np.random:
            self._rng = np.random
            self._srng = RandomStreams()
        else:
            if isinstance(random, int):
                self._rng = np.random.RandomState(random)
                self._srng = RandomStreams(seed=random)
            elif isinstance(random, np.random.RandomState):
                self._rng = random
                self._srng = RandomStreams(
                    seed=int(random.__getstate__()[1][0]))
            else:
                # handle the exception
                raise TypeError
# pylint: enable=too-few-public-methods
