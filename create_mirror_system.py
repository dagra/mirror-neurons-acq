"""Mirror system

Train a neural network and save it as the mirror system
"""
import os

import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from agent import Agent, calc_mirror_system_input, AVAILABLE_ACTIONS
from external_enviroment import ExternalEnviroment

np.set_printoptions(precision=3)


def normalize_minmax(dataset):
    x = []
    y = []
    for d in dataset:
        x.append(d[0])
        y.append(d[1])
    x = np.asarray(x)
    y = np.asarray(y)
    max_x = np.max(x)
    min_x = np.min(x)
    x = (x - min_x) / float(max_x - min_x)
    x = x * 2 - 1
    print np.max(x), np.min(x)
    y[y == 0] = -1
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
    zeros = np.zeros(4900)
    g_i = 0
    while samples_counter < int(size):
        if g_i % 5 == 0:
            agent = Agent(n_irrelevant_actions=0)
            print g_i, samples_counter, counter_per_class, counter_zero
            if g_i % 10 == 0:
                env.reset_random()
            else:
                env.reset()
        n_tried_actions = 0
        while n_tried_actions < max_actions:
            executed = agent.act(env)
            n_tried_actions += 1
            if executed:
                inp = calc_mirror_system_input(agent.current_state,
                                               agent.next_state,
                                               agent.hunger)
                action_i = np.nonzero(agent.training_signal)
                if counter_per_class[action_i] < max_per_class:
                    dataset.append([inp, agent.training_signal])
                    samples_counter += 1
                    counter_per_class[action_i] += 1
            elif counter_zero < max_zero:
                dataset.append([np.append(zeros, agent.hunger),
                                np.zeros(len(AVAILABLE_ACTIONS[:-1]))])
                samples_counter += 1
                counter_zero += 1
            if agent.hunger == 0:
                agent.hunger = 1
                break
        g_i += 1

    print "Dataset creation time: {} sec.".format(time.time() - start_time)
    return np.asarray(dataset)


class Model(nn.Module):
    def __init__(self, trndataX, trndataY):
        super(Model, self).__init__()
        # In the paper they say that they use the logsigmoid for
        # activation function. But given that the logsigmoid gives
        # values in range (-inf, 0) and the priming is before the
        # application of the function, x cannot be larger than 0
        # so there cannot be any reinforcement signal.
        self.l1 = nn.Sequential(nn.Dropout(p=0.5),
                                nn.Linear(np.size(trndataX[0], 0), 500),
                                nn.LeakyReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(500, 100),
                                nn.LeakyReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(100, 40),
                                nn.LeakyReLU(),
                                nn.Linear(40, 20),
                                nn.LeakyReLU(),
                                nn.Linear(20, np.size(trndataY[0], 0)),)
                                # nn.Tanh())

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

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
        out = net(Variable(torch.from_numpy(
            np.asarray([sample[0]])).type(FloatTensor)))
        if useCuda:
            out = out.cpu().data.numpy()[0]
        else:
            out = out.data.numpy()[0]
        # print out, sample[1]
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
        dataset = create_dataset(size=5000)
        dataset_fname = "dataset_%d.npz" % (len(dataset))
        np.savez_compressed(dataset_fname, dataset=dataset)
        print "Saved dataset as: ", dataset_fname
    np.random.shuffle(dataset)

    dataset = normalize_minmax(dataset)
    # Split datasets to train/test
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
        trainer.train(epochs=15)
    except KeyboardInterrupt:
        pass

    # Evaluate network
    print "Evaluation in test set"
    evaluate_network(net, test_ds, useCuda)
    print "---------------"
    print "Evaluation in complete  dataset"
    evaluate_network(net, dataset)

    return net

if __name__ == "__main__":
    net = create_network(dataset_fname='dataset_5000.npz')
    torch.save(net, "network")
