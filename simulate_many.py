"""Simulate with different number of irrelevant actions and gather stats"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import time

from simulate_one import simulate_one
from create_mirror_system import (Model, create_network,)

# Simulation parameters
load_model = True
useCuda = True
fname_model = 'network'
net = None
dataset_fname = 'dataset_5000.npz'
if load_model:
    net = torch.load(fname_model)
elif use_mirror_system:
    net = create_network(dataset_fname=dataset_fname)
    torch.save(net, fname_model)

max_eat = 100
max_actions = 50

n_irrelevant_actions = [0, 1, 5, 10, 25, 40, 50, 70]
n_sims_per_level = 3
ms_recovery_time_per_level = []
ms_recovery_error_per_level = []
no_ms_recovery_time_per_level = []
no_ms_recovery_error_per_level = []

for n_irr_action in n_irrelevant_actions:
    no_ms_temp = []
    ms_temp = []
    for i in range(n_sims_per_level):
        print ("Running without mirror system "
               "[Irrelevant actions={}-n={}/{}]...").format(
            n_irr_action, i + 1, n_sims_per_level)
        start_time = time.time()
        recovery_time = simulate_one(max_eat, max_actions, False,
                                     n_irr_action, mirror_system=net,
                                     useCuda=useCuda, verbose=0)
        print "Time needed: {}".format(time.time() - start_time)
        no_ms_temp.append(recovery_time)

        print ("Running with mirror system "
               "[Irrelevant actions={}-n={}/{}]...").format(
            n_irr_action, i + 1, n_sims_per_level)
        start_time = time.time()
        recovery_time = simulate_one(max_eat, max_actions, True,
                                     n_irr_action, mirror_system=net,
                                     useCuda=useCuda, verbose=0)
        print "Time needed: {}".format(time.time() - start_time)
        ms_temp.append(recovery_time)
    no_ms_recovery_time_per_level.append(np.mean(no_ms_temp))
    no_ms_recovery_error_per_level.append(np.std(no_ms_temp))
    ms_recovery_time_per_level.append(np.mean(ms_temp))
    ms_recovery_error_per_level.append(np.std(ms_temp))

np.savez_compressed(
    'many_stats.npz',
    no_ms_recovery_time_per_level=no_ms_recovery_time_per_level,
    no_ms_recovery_error_per_level=no_ms_recovery_error_per_level)

plt.figure()
plt.title("Recovery time as a function of irrelevant actions")
plt.errorbar(n_irrelevant_actions, ms_recovery_time_per_level,
             ms_recovery_error_per_level, color='black', ecolor='grey',
             capsize=5, linestyle='-', label="With mirror system")
plt.errorbar(n_irrelevant_actions, no_ms_recovery_time_per_level,
             no_ms_recovery_error_per_level, color='black', ecolor='grey',
             capsize=5, linestyle=':', label="Without mirror system")
plt.legend()

plt.show()
