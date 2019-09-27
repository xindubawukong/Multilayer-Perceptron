import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


# f(x) = max(0, x)
# f'(x) = f(x)
class Relu(Layer):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, input):
        f = (input > 0) * input
        self.f = f
        return f

    def backward(self, grad_output):
        f = self.f
        return grad_output * f


# f(x) = 1 / (1 + e^(-x))
# f'(x) = f(x) * (1 - f(x))
class Sigmoid(Layer):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, input):
        f = 1.0 / (np.exp(-1 * input) + 1)
        self.f = f
        return f

    def backward(self, grad_output):
        f = self.f
        return grad_output * f * (1 - f)


class Softmax(Layer):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, input):
        e = np.exp(input)
        self.e = e
        return e / e.sum(axis=1)[:, None]

    def backward(self, grad_output):
        e = self.e
        sum = e.sum(axis=1)[:, None]
        tmp = grad_output * -e / sum**2
        grad = tmp - grad_output * 2 * -(e/sum)**2 + grad_output * (e / sum - (e / sum)**2)
        return grad


# y = x.dot(W) + b
class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super().__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        self._saved_for_backward(input)
        return input.dot(self.W) + self.b

    def backward(self, grad_output):
        x = self._saved_tensor
        grad_y = grad_output
        batch_size = grad_y.shape[0]
        self.grad_W = x.T.dot(grad_y) / batch_size
        self.grad_b = grad_y.T.sum(axis=1) / batch_size
        return grad_y.dot(self.W.T)

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b


if __name__ == '__main__':
    layer1 = Softmax('softmax')
    a = np.array([[-1, 0],
                  [0, 1]])
    b = np.array([[-1, 1],
                   [1, -1]])
    print(layer1.forward(a))
    print(layer1.backward(np.array(b)))
