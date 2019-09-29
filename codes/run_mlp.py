from network import Network
from utils import *
from layers import Relu, Sigmoid, Softmax, Linear
from loss import EuclideanLoss, CrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import matplotlib.pyplot as plt


train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Linear('fc1', 784, 128, 0.01))
model.add(Relu('relu1'))
model.add(Linear('fc2', 128, 128, 0.01))
model.add(Relu('relu2'))
model.add(Linear('fc3', 128, 10, 0.01))
model.add(Softmax('softmax'))

# loss = EuclideanLoss(name='loss')
loss = CrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.1,
    'weight_decay': 0.0,
    'momentum': 0.0,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 100,
    'test_epoch': 5
}

reset_time()
record = []
cnt = 0
for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    epoch_record = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'], test_data, test_label)
    min_loss = min(x[1] for x in epoch_record)
    if len(record) == 0 or min_loss < min(x[2] for x in record):
        cnt = 0
    else:
        cnt += 1
    record += [(epoch, x[0], x[1], x[2], x[3]) for x in epoch_record]
    if cnt == 3:
        break

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)
max_id = max(t[1] for t in record)
x = [t[0] * max_id + t[1] for t in record]
y = [t[2] for t in record]
line1 = ax.plot(x, y, 'r', label='loss')
ax2 = ax.twinx()
y = [t[3] for t in record]
line2 = ax2.plot(x, y, label='accuracy')

line = line1 + line2
labs = [l.get_label() for l in line]
ax.legend(line, labs, loc=0)

fig.savefig('temp.png')

print('best accuracy:', max(x[3] for x in record))
print('time:', record[-1][4])
