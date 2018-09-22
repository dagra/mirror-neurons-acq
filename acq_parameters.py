"""Contains the general parameters regarding ACQ"""

import numpy as np


class ACQparameters:
    # Population code width
    # Allows reinforcement to effect similar states
    # In the paper s_p = 0.25
    s_p = 0.4

    # Executability noise
    # Encourages exploration, but would not override actions with
    # priority greater than 0.25
    # In the paper uniformly in [0, 0.25]
    @classmethod
    def e_e(cls):
        return np.random.uniform(low=0, high=0.25)

    # Desirability noise
    # Encourages exploration, but would not override actions with
    # priority greater than 0.25
    # In the paper uniformly in [0, 0.25]
    @classmethod
    def e_d(cls):
        return np.random.uniform(low=0, high=0.25)

    # Efference copy decay rate
    # Set to 10% of maximal mirror neuron activation to yield
    # priming effect
    # In the paper k = 0.1
    k = 0.1

    # Executability decrease threshold
    # Ensures executability is only decreased if the mirror system
    # is not activated at 25% of its maximal level
    # (needs to be greater than k)
    # In the paper psi = 0.25
    psi = 0.15

    # Executability/desirability learning rate
    # Determines rate of weight changes -- the model becomes
    # unstable when this value is too large
    # In the paper a = 0.1
    # The model is very sensitive of this parameter
    a = 0.05

    # Desirability discount rate
    # Determines maximal length of action sequences that can be
    # learned
    # In the paper gamma = 0.9
    gamma = 0.9
