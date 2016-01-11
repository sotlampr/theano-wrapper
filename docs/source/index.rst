Welcome to theano-wrapper's documentation!
==========================================

Theano-wrapper is a library for machine learning built around
`theano <http://deeplearning.net/software/theano/>`_ and implementing the
deep learning tutorials found `here <http://deeplearning.net/tutorial/>`_
in python 3.

Requirements
------------

`numpy <https://github.com/numpy/numpy>`_, `scipy <https://github.com/scipy/scipy>`_,
`theano <https://github.com/Theano/Theano>`_ for computations

`nose <https://github.com/nose-devs/nose/>`_,
`coverage <https://pypi.python.org/pypi/coverage>`_ for testing

`scikit-learn <https://github.com/scikit-learn/scikit-learn>`_ for some helpful utilities

Installation & first steps
--------------------------

Setup your virtual environment as you like, navigate to a temp directory
and execute::

    git clone https://github.com/sotlampr/theano-wrapper
    cd theano-wrapper
    pip install requirements.txt
    pip install -e ./


For a demo, open a python interpreter and type:

    >>> from theano_wrapper.demo import demo
    >>> demo()

for more information, visit `Tutorial <tutorial>`_  and `Reference <reference>`_ section.

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorial
   reference
   license
   help

* :ref:`modindex`
* :ref:`search`

