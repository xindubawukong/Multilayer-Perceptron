from utils import *
import numpy as np


def data_iterator(x, y, batch_size, shuffle=True):
    indx = list(range(len(x)))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]]


def train_net(model, loss, config, train_inputs, train_labels, batch_size, disp_freq, test_input, test_labels):

    iter_counter = 0
    loss_list = []
    acc_list = []

    record = []

    for input, label in data_iterator(train_inputs, train_labels, batch_size):
    # for input, label in [(inputs[:2], labels[:2]) for i in range(100)]:
        target = onehot_encoding(label, 10)
        iter_counter += 1

        # forward net
        output = model.forward(input)
        # calculate loss
        loss_value = loss.forward(output, target)
        # generate gradient w.r.t loss
        grad = loss.backward(output, target)
        # backward gradient

        model.backward(grad)
        # update layers' weights
        model.update(config)

        # print('lanel:', label)
        # print('output:', output)
        # print('loss_value:', loss_value)
        # print('grad:', grad)

        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)

        if iter_counter % disp_freq == 0:
            msg = '  Training iter %d, batch loss %.4f, batch acc %.4f' % (iter_counter, np.mean(loss_list), np.mean(acc_list))
            loss_list = []
            acc_list = []
            LOG_INFO(msg)

            test_loss, test_acc = test_net(model, loss, test_input, test_labels, batch_size)
            record.append((iter_counter, test_loss, test_acc, get_run_time()))
    
    return record


def test_net(model, loss, inputs, labels, batch_size):
    loss_list = []
    acc_list = []

    for input, label in data_iterator(inputs, labels, batch_size, shuffle=False):
        target = onehot_encoding(label, 10)
        output = model.forward(input)
        loss_value = loss.forward(output, target)
        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)

    msg = '    Testing, total mean loss %.5f, total acc %.5f' % (np.mean(loss_list), np.mean(acc_list))
    LOG_INFO(msg)

    return np.mean(loss_list), np.mean(acc_list)
