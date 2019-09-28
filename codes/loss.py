from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        e = 0.5 * (input - target) ** 2
        return e.sum(axis=1)


    def backward(self, input, target):
        a = input * (target == 0)
        b = (input - 1) * (target == 1)
        return a + b


class CrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        e = -1 * target * np.log(input)
        return e.sum(axis=1)

    def backward(self, input, target):
        return -1 * target / input


if __name__ == '__main__':
    input = np.array([[0.5, 0.4, 0.1],
                      [0.1, 0.3, 0.6]])
    target = np.array([[1, 0, 0],
                       [0, 1, 0]])

    print('EuclideanLoss:')
    loss1 = EuclideanLoss('loss1')
    print(loss1.forward(input, target))
    print(loss1.backward(input, target))

    print('\nCrossEntropyLoss:')
    loss1 = CrossEntropyLoss('loss2')
    print(loss1.forward(input, target))
    print(loss1.backward(input, target))
