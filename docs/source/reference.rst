Reference
=========

Layers
------

Linear Regression
~~~~~~~~~~~~~~~~
.. autoclass:: theano_wrapper.layers.LinearRegression

Logistic Regression
~~~~~~~~~~~~~~~~~~~
.. autoclass:: theano_wrapper.layers.LogisticRegression

Trainers
--------
Epoch-based
~~~~~~~~~~~
.. autoclass:: theano_wrapper.trainers.EpochTrainer(clf, alpha=0.01, max_iter=10000, patience=5000, p_inc=2.0, imp_thresh=0.995, random=None, verbose=None)

Stohastic Gradient Descent
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: theano_wrapper.trainers.SGDTrainer(clf, batch_size=None, alpha=0.01, max_iter=10000, patience=5000, p_inc=2.0, imp_thresh=0.995, random=None, verbose=None)
