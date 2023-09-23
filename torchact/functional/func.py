from torchact.nn import *


def ab_relu(x):
    r"""
    Implementation of Average-Biased Rectified Linear Unit.
    """
    m = ABReLU()
    return m(x)


def elu(x):
    r"""
    Implementation of Exponential Linear Unit.
    """
    m = ELU()
    return m(x)


def leaky_relu(x):
    r"""
    Implementation of LeakyReLU.
    """
    m = LeakyReLU()
    return m(x)


def log_sigmoid(x):
    r"""
    Implementation of Log(Sigmoid).
    """
    m = LogSigmoid()
    return m(x)


def log_softmax(x):
    r"""
    Implementation of Log(Softmax(x)).
    """
    m = LogSoftmax()
    return m(x)


def relu(x):
    r"""
    Implementation of Rectified Linear Unit.
    """
    m = ReLU()
    return m(x)


def sigmoid(x):
    r"""
    Implementation of Sigmoid.
    """
    m = Sigmoid()
    return m(x)


def sinlu(x):
    r"""
    Implementation of Sinu-Sigmoidal Linear Unit
    """
    m = SinLU()
    return m(x)


def softmax(x):
    r"""
    Implementation of Softmax.
    """
    m = Softmax()
    return m(x)


def step(x):
    r"""
    Implementation of binary step activation.
    """
    m = Step()
    return m(x)
