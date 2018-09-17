"""This file contains the state of the external enviroment."""
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from acq_parameters import ACQparameters as ACQprms


class ExternalEnviroment:
    """Contains all the info about the external enviroment"""

    def __init__(self, v_max=35, random=False):
        self.v_max = v_max
        # Create constants for optimization
        self.outer_divisor = ((ACQprms.s_p ** 2) * 2 * np.pi)
        self.inner_divisor = 2 * (ACQprms.s_p**2)
        self.x = np.repeat(np.arange(v_max)[..., np.newaxis], v_max, axis=1)
        self.y = np.repeat(np.arange(v_max)[..., np.newaxis], v_max, axis=1).T

        if random:
            self.rest_random()
        else:
            self.reset()

    def reset(self):
        self.food = np.asarray([30, 30])
        self.paw = np.asarray([0, 0])
        self.mouth = np.asarray([0, self.v_max])
        self.tube = np.asarray([25, 30])
        self.compute_population_codes()

    def reset_random(self):
        self.food = np.asarray([np.random.randint(0, self.v_max - 1),
                                np.random.randint(0, self.v_max - 1)])
        self.paw = np.asarray([np.random.randint(0, self.v_max - 1),
                               np.random.randint(0, self.v_max - 1)])
        self.mouth = np.asarray([np.random.randint(0, self.v_max - 1),
                                np.random.randint(0, self.v_max - 1)])
        self.tube = np.asarray([np.random.randint(0, self.v_max - 1),
                                np.random.randint(0, self.v_max - 1)])
        self.compute_population_codes()

    def population_code(self, v1, v2):
        dx, dy = v2[0] - v1[0], v2[1] - v1[1]
        norm = int(self.v_max / 2)
        code = np.zeros((self.v_max, self.v_max)).astype('float')
        # code1 = np.zeros((self.v_max, self.v_max)).astype('float')
        # for x in range(code.shape[0]):
        #     for y in range(code.shape[1]):
        #         code1[x, y] = np.exp((-(dx - (x - norm))**2 -
        #                             (dy - (y - norm))**2) /(
        #                             2 * ACQprms.s_p**2))
        x = self.x
        y = self.y
        # print x, y
        # print dx, dy, norm, self.inner_divisor,
        code = np.exp((-(dx - (x - norm))**2 - (dy - (y - norm))**2) /
                      self.inner_divisor).astype('float')
        # print code > 0.01
        # print code1
        # print np.all(code1 == code), np.nonzero(code1 == code)
        return code / self.outer_divisor

    def compute_population_codes(self):
        self.pf = self.population_code(self.paw, self.food)
        self.mf = self.population_code(self.mouth, self.food)
        self.bf = self.population_code(self.tube, self.food)
        self.pb = self.population_code(self.paw, self.tube)

    def get_population_codes(self):
        d = OrderedDict()
        d['pf'] = self.pf[:]
        d['mf'] = self.mf[:]
        d['bf'] = self.bf[:]
        d['pb'] = self.pb[:]
        return d

    def visualize_population_codes(self):
        plt.figure()
        t = "paw = (%d, %d)" % (self.paw[0], self.paw[1])
        t = t + ", " + "food = (%d, %d)" % (self.food[0], self.food[1])
        dx, dy = self.paw[0] - self.food[0], self.paw[1] - self.food[1]
        t = t + ", " + "(dx, dy) = (%d, %d)" % (dx, dy)
        plt.title(t)
        plt.imshow(self.pf.T, origin="lower")
        plt.colorbar()

        plt.figure()
        t = "mouth = (%d, %d)" % (self.mouth[0], self.mouth[1])
        t = t + ", " "food = (%d, %d)" % (self.food[0], self.food[1])
        dx, dy = self.mouth[0] - self.food[0], self.mouth[1] - self.food[1]
        t = t + ", " + "(dx, dy) = (%d, %d)" % (dx, dy)
        plt.title(t)
        plt.imshow(self.mf.T, origin="upper")
        plt.colorbar()

        plt.figure()
        t = "tube = (%d, %d)" % (self.tube[0], self.tube[1])
        t = t + ", " "food = (%d, %d)" % (self.food[0], self.food[1])
        dx, dy = self.tube[0] - self.food[0], self.tube[1] - self.food[1]
        t = t + ", " + "(dx, dy) = (%d, %d)" % (dx, dy)
        plt.title(t)
        plt.imshow(self.bf.T, origin="lower")
        plt.colorbar()

        plt.figure()
        t = "paw = (%d, %d)" % (self.paw[0], self.paw[1])
        t = t + ", " "tube = (%d, %d)" % (self.tube[0], self.tube[1])
        dx, dy = self.paw[0] - self.tube[0], self.paw[1] - self.tube[1]
        t = t + ", " + "(dx, dy) = (%d, %d)" % (dx, dy)
        plt.title(t)
        plt.imshow(self.pb.T, origin="lower")
        plt.colorbar()

        plt.show()

    def print_current_state(self):
        print "Food: {}, Mouth: {}, Paw: {}, Tube: {}".format(self.food,
                                                              self.mouth,
                                                              self.paw,
                                                              self.tube)
