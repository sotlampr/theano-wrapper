[![Build Status](https://travis-ci.org/sotlampr/theano-wrapper.svg?branch=master)](https://travis-ci.org/sotlampr/theano-wrapper)
[![Coverage Status](https://coveralls.io/repos/sotlampr/theano-wrapper/badge.svg?branch=master&service=github)](https://coveralls.io/github/sotlampr/theano-wrapper?branch=master)
[![Documentation Status](https://readthedocs.org/projects/theano-wrapper/badge/?version=latest)](http://theano-wrapper.readthedocs.org/en/latest/?badge=latest)
Project no longer developed, I am switching to [tensorflow](https://github.com/tensorflow/tensorflow)

---

# theano-wrapper
*Neural network library based on theano*

## Goal
The goal of this project is to cover all the material of the official [Theano deep learning tutorial](http://deeplearning.net/tutorial/)
and implement the appropriate classes and functions in Python 3.

## Requirements

**[numpy](https://github.com/numpy/numpy), [scipy](https://github.com/scipy/scipy), [theano](https://github.com/Theano/Theano)** for computations

**[nose](https://github.com/nose-devs/nose/), [coverage](https://pypi.python.org/pypi/coverage)** for testing

**[scikit-learn](https://github.com/scikit-learn/scikit-learn)** for some helpful utilities

## Installation

Setup your virtual environment as you like, navigate to a temp directory and execute:

    git clone https://github.com/sotlampr/theano-wrapper
    cd theano-wrapper
    pip install requirements.txt
    pip install -e ./


## Usage

For a demo, open a python interpreter and type:

    >>> from theano_wrapper.demo import demo
    >>> demo()

For a complete documentation visit the [read the docs](http://theano-wrapper.readthedocs.org/en/latest/index.html) page.

## What is included

* Regression estimators
    * Linear Regression
    * Multilayer Linear Regression

* Classification estimators
    * Logistic Regression
    * Multilayer Perceptron

* Unsupervised
    * Single hidden layer tied autoencoder
    * Denoising autoencoder

* Training Classes
    * Simple epoch-based gradient descent training
    * (Minibatch) Stohastic gradient descent training

* Regularization
    * L1 and L2 squared

* \> 95% testing coverage

## What is not included
* More Estimators and trainers [coming soon]
* Documentation
* Error handling
* Testing for extreme cases

## Contributing

TODO
