from theano import tensor as T


class DummyClf:
    X = T.matrix('X')
    y = T.ivector('y')
    W = T.matrix('W')
    b = T.vector('b')

    params = [W, b]
