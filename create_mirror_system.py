"""Mirror system

Train a neural network and save it as the mirror system
"""
import os
from sets import Set

import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from agent import Agent, calc_mirror_system_input, AVAILABLE_ACTIONS
from external_enviroment import ExternalEnviroment

np.set_printoptions(precision=3)


def check_dataset(dataset):
    print "Checking dataset for inconsistent elements..."
    inconsistent = Set([])
    inds = []
    for i in range(len(dataset)):
        if np.all(dataset[i][1] == 0):
            continue
        inds.append(i)
    start_time = time.time()
    print len(inds)
    while inds:
        i = inds.pop()
        if i % 200 == 0:
            print i, time.time() - start_time
            start_time = time.time()
        x = dataset[i][0]
        y = dataset[i][1]
        temp = []
        for j in inds[:]:
            if j in inconsistent:
                continue
            if np.all(dataset[j][0] == x) and not np.all(dataset[j][1] == y):
                temp.append(j)
        if temp:
            inconsistent.update(set(temp))
            inconsistent.add(i)

    # for i in range(len(dataset)):
    #     x = dataset[i][0]
    #     y = dataset[i][1]
    #     if np.all(y == 0):
    #         continue
    #     temp = []
    #     for j in range(i, len(dataset)):
    #         if j in inconsistent:
    #             continue
    #         if np.all(dataset[j][0] == x) and not np.all(dataset[j][1] == y):
    #             temp.append(j)
    #     if temp:
    #         inconsistent.update(set(temp))
    #         inconsistent.add(i)
    return inconsistent


def preprocess_dataset(dataset):
    print "Preprocessing dataset..."
    x = []
    y = []
    for d in dataset:
        x.append(d[0])
        y.append(d[1])
    x = np.asarray(x)
    y = np.asarray(y)
    # Normalize minmax
    # max_x = np.max(x)
    # min_x = np.min(x)
    # x = (x - min_x) / float(max_x - min_x)
    # x = x * 2 - 1
    # x[:, -1] = x[:, -1] * 100
    # Set target 0s to -1
    y[y == 0] = -0.25
    d = []
    for i in range(len(dataset)):
        d.append([x[i], y[i]])
    return np.asarray(d)


def create_dataset(size=5000):
    start_time = time.time()
    env = ExternalEnviroment()
    agent = Agent(n_irrelevant_actions=0)
    dataset = []
    samples_counter = 0
    max_actions = 50
    counter_per_class = np.zeros(len(AVAILABLE_ACTIONS[:-1]))
    max_per_class = int(size / (len(counter_per_class) + 1))
    counter_zero = 0
    max_zero = max_per_class
    zeros = np.zeros(140)
    g_i = 0
    while samples_counter < int(size):
        if g_i % 5 == 0:
            agent = Agent(n_irrelevant_actions=0)
            print g_i, samples_counter, counter_per_class, counter_zero
            if g_i % 25 == 0:
                env.reset()
            else:
                env.reset_random()
        n_tried_actions = 0
        while n_tried_actions < max_actions:
            executed = agent.act(env)
            n_tried_actions += 1
            if executed:
                inp = calc_mirror_system_input(agent.current_state,
                                               agent.next_state,
                                               agent.hunger)
                action_i = np.nonzero(agent.training_signal)
                if counter_per_class[action_i] < max_per_class and \
                   ((not np.all(inp[:-1] == 0)) or inp[-1] == 1):
                    dataset.append([inp, agent.training_signal])
                    samples_counter += 1
                    counter_per_class[action_i] += 1
            elif counter_zero < max_zero:
                dataset.append([np.append(zeros, 0),
                                np.zeros(len(AVAILABLE_ACTIONS[:-1]))])
                samples_counter += 1
                counter_zero += 1
            if agent.hunger == 0:
                agent.hunger = 1
                env.reset()
                break
        g_i += 1
    for i in range(max_zero * 9):
        dataset.append([np.append(zeros, 0),
                        np.zeros(len(AVAILABLE_ACTIONS[:-1]))])
    print "Dataset creation time: {} sec.".format(time.time() - start_time)
    dataset = np.asarray(dataset)
    inconsistent = check_dataset(dataset)
    print "Found %d inconsistent data" % len(inconsistent)
    for i in inconsistent:
        dataset = np.delete(dataset, i, axis=0)
    return np.asarray(dataset)


class Model(nn.Module):
    def __init__(self, trndataX, trndataY):
        super(Model, self).__init__()
        # In the paper they say that they use the logsigmoid for
        # activation function. But given that the logsigmoid gives
        # values in range (-inf, 0) and the priming is before the
        # application of the function, x cannot be larger than 0
        # so there cannot be any reinforcement signal.
        print trndataX[0].shape
        self.l1 = nn.Sequential(nn.Linear(np.size(trndataX[0], 0), 300),
                                nn.LeakyReLU(),
                                # nn.Dropout(p=0.5),
                                nn.Linear(300, 150),
                                nn.LeakyReLU(),
                                # nn.Dropout(p=0.5),
                                nn.Linear(150, 50),
                                nn.LeakyReLU(),
                                nn.Linear(50, 50),
                                nn.LeakyReLU(),
                                nn.Linear(50, np.size(trndataY[0], 0)),)

    def forward(self, x):
        out = self.l1(x)
        return out


class Trainer:
    def __init__(self, model, trndataX, trndataY, useCuda):
        self.model = model
        if useCuda:
            self.model.cuda()
        self.FloatTensor = \
            torch.cuda.FloatTensor if useCuda else torch.FloatTensor
        self.useCuda = useCuda

        # Select criterion
        self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()

        # Select optimizer
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr,
        #                                      alpha=0.99, weight_decay=0.3,
        #                                      momentum=momentum)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001,
        #                                  momentum=0.9)
        # self.optimizer = optim.Adadelta(self.model.parameters())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001,
                                    weight_decay=0.00)

        self.trndataX, self.trndataY = trndataX, trndataY
        self.trndata_size = len(self.trndataX)

    def train(self, epochs=1):
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            epoch_time = time.time()
            epoch_loss = 0
            random_perm = np.random.permutation(self.trndata_size)
            self.trndataX = self.trndataX[random_perm]
            self.trndataY = self.trndataY[random_perm]
            for i in range(len(self.trndataX)):
                inputs = Variable(torch.from_numpy(self.trndataX[i]).type(
                    self.FloatTensor))
                targets = Variable(torch.from_numpy(self.trndataY[i]).type(
                    self.FloatTensor))

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data[0]

            print ("->Epoch [{}/{}], Epoch time: {} sec ({} min), Total time:"
                   "{}sec ({} min), Epoch loss = {}").format(
                epoch, epochs, (time.time() - epoch_time),
                int((time.time() - epoch_time) / 60.),
                int(time.time() - start_time),
                int((time.time() - start_time) / 60.),
                epoch_loss / len(self.trndataX))


def evaluate_network(net, dataset, useCuda=False):
    FloatTensor = torch.cuda.FloatTensor if useCuda else torch.FloatTensor
    correct = 0
    partial_correct = 0
    correct_per_class = np.zeros(len(dataset[0, 1]))
    true_positive = np.zeros(len(dataset[0, 1]))
    true_negative = np.zeros(len(dataset[0, 1]))
    false_positive = np.zeros(len(dataset[0, 1]))
    false_negative = np.zeros(len(dataset[0, 1]))
    counter_per_class = np.zeros(len(dataset[0, 1]))
    partial_correct_per_class = np.zeros(len(dataset[0, 1]))
    for sample in dataset:
        if np.all(sample[1] <= 0):
            continue
        out = net(Variable(torch.from_numpy(
            np.asarray([sample[0]])).type(FloatTensor)))
        if useCuda:
            out = out.cpu().data.numpy()[0]
        else:
            out = out.data.numpy()[0]
        out = np.where(out == np.amax(out))[0]
        true_out = np.argmax(sample[1])
        if len(out) == 1 and out[0] == true_out:
            correct += 1
            true_positive[true_out] += 1
            partial_correct_per_class[true_out] += 1
            true_negative += 1
            true_negative[true_out] -= 1
            correct_per_class[true_out] += 1
        else:
            false_positive[true_out] += 1
            false_negative[true_out] += 1
            if len(out) > 1 and true_out in out:
                print out, true_out
                partial_correct += 1
                partial_correct_per_class[true_out] += 1
        counter_per_class[true_out] += 1

    print "Correct rate: {}".format(float(correct) / float(len(dataset)))
    print "Partial Correct rate: {}".format(float(partial_correct) /
                                            float(len(dataset)))
    print "True positive: {}".format(true_positive.astype('float') /
                                     (true_positive + false_negative))
    print "False positive: {}".format(false_positive.astype('float') /
                                      (true_negative + false_positive))
    print "True negative: {}".format(true_negative.astype('float') /
                                     (true_negative + false_positive))
    print "False negative: {}".format(false_negative.astype('float') /
                                      (true_positive + false_negative))
    print "Examples per class: ", counter_per_class
    print "Correct per class: {}".format(correct_per_class.astype('float') /
                                         counter_per_class)
    print "Partial Correct per class: {}".format(
        partial_correct_per_class.astype('float') /
        counter_per_class)
    print "Actions: ", map(lambda x: x().name, AVAILABLE_ACTIONS[:-1])


def create_network(train_prc=0.8, dataset_fname=None, useCuda=False):
    if train_prc <= 0. or train_prc >= 1:
        raise ValueError("Train percentile must be in range (0,1)")
    # Create dataset
    if dataset_fname and os.path.isfile(dataset_fname):
        dataset = np.load(dataset_fname)['dataset']
    else:
        dataset = create_dataset(size=15000)
        dataset_fname = "dataset_%d.npz" % (len(dataset))
        np.savez_compressed(dataset_fname, dataset=dataset)
        print "Saved dataset as: ", dataset_fname

    dataset = preprocess_dataset(dataset)
    # Split datasets to train/test
    np.random.shuffle(dataset)
    size_train_ds = int(len(dataset) * train_prc)
    size_test_ds = len(dataset) - size_train_ds
    train_ds = dataset[:size_train_ds]
    test_ds = dataset[size_train_ds:]

    print "Train dataset size: ", size_train_ds
    print "Test dataset size: ", size_test_ds
    print "Sample input size: X={}, Y={}".format(train_ds[0, 0].shape,
                                                 train_ds[0, 1].shape)
    print "Sample input: X={}, Y={}".format(train_ds[0, 0], train_ds[0, 1])
    print "Sample input X, nonzero: ", np.nonzero(train_ds[100, 0])[0]
    print "Examples per class in train set: ", train_ds[:, 1].shape
    print "Examples per class in test set: ", test_ds[:, 1].shape
    # Create neural network model
    net = Model(train_ds[:, 0], train_ds[:, 1])
    # Create trainer for the model
    trainer = Trainer(net, train_ds[:, 0], train_ds[:, 1], useCuda)
    # Train the model
    try:
        trainer.train(epochs=100)
    except KeyboardInterrupt:
        pass

    # Evaluate network
    print "Evaluation in test set"
    evaluate_network(net, test_ds, useCuda)
    print "---------------"
    print "Evaluation in complete dataset"
    evaluate_network(net, dataset, useCuda)

    return net

if __name__ == "__main__":
    useCuda = True
    net = create_network(dataset_fname='dataset_9500.npz', useCuda=useCuda)
    torch.save(net, "network")
