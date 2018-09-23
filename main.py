"""Main code

Initializes the agent and the enviroment.
Simulates N trials.
Records and plots the corresponding statistics.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agent
from external_enviroment import ExternalEnviroment
from create_mirror_system import (Model, create_network,
                                  )

# Use mirror system
use_mirror_system = False
load_model = True
useCuda = True
fname_model = 'network'
net = None
dataset_fname = 'dataset_5000.npz'
if use_mirror_system and load_model:
    net = torch.load(fname_model)
elif use_mirror_system:
    net = create_network(dataset_fname=dataset_fname)
    torch.save(net, fname_model)

# Initialization
env = ExternalEnviroment()

agent = Agent(use_mirror_system=use_mirror_system,
              n_irrelevant_actions=25, mirror_system=net,
              useCuda=useCuda)

max_eat = 100
max_actions = 50
max_eat = 2 * max_eat

n_rel_actions = agent.n_actions - agent.n_irr_actions
# Statistics
executability_error = np.zeros(max_eat) - 1
success_trial_length = np.zeros(max_eat) - 1
total_trial_length = []
total_executability_error = []
last_action_desirability = np.zeros((n_rel_actions, max_eat)) - 1
all_last_action_desirability = np.zeros((n_rel_actions, max_eat)) - 1
i_all_last = 0
trial = 0
trial_success = 0
# A trial is successful if the agent performs the task (eat)
try:
    while trial_success < max_eat / 2:
        n_tried_actions = 0
        start_time = time.time()
        executability_error[trial_success] = 0
        total_executability_error.append(0)
        while n_tried_actions < max_actions:
            executed = agent.act(env)
            executability_error[trial_success] += int(not executed)
            total_executability_error[trial] += int(not executed)
            n_tried_actions += 1
            if agent.hunger == 0:
                last_action_desirability[:, trial_success] = \
                    agent.hist_desirability[:n_rel_actions, -1]
                executability_error[trial_success] /= float(n_tried_actions)
                success_trial_length[trial_success] = n_tried_actions
                trial_success += 1
                break
        if i_all_last >= all_last_action_desirability.shape[-1]:
            all_last_action_desirability = np.append(
                all_last_action_desirability,
                np.zeros((n_rel_actions, max_eat)) - 1,
                axis=1)
        all_last_action_desirability[:, i_all_last] = \
            agent.hist_desirability[:n_rel_actions, -1]
        i_all_last += 1
        total_executability_error[trial] /= float(n_tried_actions)
        total_trial_length.append(n_tried_actions)

        print "###########"
        print "[{}]: {}/{}-Ate: {}-actions:{}-exec rate: {}-time:{} sec".format(
            trial + 1, trial_success, max_eat, agent.hunger == 0,
            n_tried_actions, total_executability_error[trial],
            np.round(time.time() -
                     start_time))
        # Debug
        print agent.action_counter[:agent.n_rel_actions]

        agent.action_counter = np.zeros(agent.n_actions)
        trial += 1

        # Reset agent and enviroment
        agent.hunger = 1
        env.reset()
except KeyboardInterrupt:
    pass

agent.apply_lesion()
lesion_time = [trial_success, i_all_last]

try:
    recovery_time = [0, 0, 0]
    recovery_flag = 0
    while trial_success < max_eat:
        n_tried_actions = 0
        start_time = time.time()
        executability_error[trial_success] = 0
        total_executability_error.append(0)
        while n_tried_actions < max_actions:
            executed = agent.act(env)
            executability_error[trial_success] += int(not executed)
            total_executability_error[trial] += int(not executed)
            n_tried_actions += 1
            if agent.hunger == 0:
                last_action_desirability[:, trial_success] = \
                    agent.hist_desirability[:n_rel_actions, -1]
                executability_error[trial_success] /= float(n_tried_actions)
                success_trial_length[trial_success] = n_tried_actions
                trial_success += 1
                break
        if i_all_last >= all_last_action_desirability.shape[-1]:
            all_last_action_desirability = np.append(
                all_last_action_desirability,
                np.zeros((n_rel_actions, max_eat)) - 1,
                axis=-1)
        all_last_action_desirability[:, i_all_last] = \
            agent.hist_desirability[:n_rel_actions, -1]
        i_all_last += 1

        total_executability_error[trial] /= float(n_tried_actions)
        total_trial_length.append(n_tried_actions)
        # Check recovery
        if agent.hunger == 0 and agent.action_counter[6] >= 1 and \
           agent.action_counter[7] >= 1 and agent.w_is[6] > 0.05 and \
           recovery_time[1] == 0:
            recovery_flag += 1
            recovery_time[0] += 1
            if recovery_flag == 3:
                recovery_time[1] = trial_success - 1
                recovery_time[2] = i_all_last - 1
        elif recovery_time[1] == 0:
            recovery_flag = 0
            recovery_time[0] += 1
        print "###########"
        print "[{}]: {}/{}-Ate: {}-actions:{}-exec rate: {}-time:{} sec".format(
            trial + 1, trial_success, max_eat, agent.hunger == 0,
            n_tried_actions, total_executability_error[trial],
            np.round(time.time() -
                     start_time))
        # Debug
        print agent.action_counter

        agent.action_counter = np.zeros(agent.n_actions)
        trial += 1

        # Reset agent and enviroment
        agent.hunger = 1
        env.reset()
except KeyboardInterrupt:
    pass

print "Trials needed for recovery: %d" % recovery_time[0], recovery_time
# Plot results

labels = map(lambda x: x.name, agent.actions.values()[:n_rel_actions])
colors = map(lambda x: x.color, agent.actions.values()[:n_rel_actions])
markers = map(lambda x: x.marker, agent.actions.values()[:n_rel_actions])

# plt.figure()
# plt.title("Desirability per action of all steps in the simulation")
# for i in range(n_rel_actions):
#     plt.plot(agent.hist_desirability[i, :], label=labels[i], color=colors[i],
#              marker=markers[i], mfc='none')
# plt.legend()

plt.figure()
plt.title("Desirability per action in the last step of all trials")
for i in range(n_rel_actions):
    plt.plot(all_last_action_desirability[i, :i_all_last], label=labels[i], color=colors[i],
             marker=markers[i], mfc='none')
plt.axvline(x=recovery_time[2], color='black', ls='--', label='Recovery')
plt.axvline(x=lesion_time[1], color='black', label='Lesion')
plt.legend()

plt.figure()
plt.title("Desirability per action in the last step of the successful trials")
for i in range(n_rel_actions):
    plt.plot(last_action_desirability[i, :], label=labels[i], color=colors[i],
             marker=markers[i], mfc='none')
plt.axvline(x=recovery_time[1], color='black', ls='--', label='Recovery')
plt.axvline(x=lesion_time[0], color='black', label='Lesion')
plt.legend()

# plt.figure()
# plt.title("Executability error of successful trials")
# plt.plot(executability_error, 'o', marker='*',
#          label='Mean executability error')
# plt.legend()

# plt.figure()
# plt.title("Executability error of all trials")
# plt.plot(total_executability_error, 'o', marker='*',
#          label='Mean executability error of all trials')
# plt.legend()

# plt.figure()
# plt.title("Trial length of successful trials")
# plt.plot(success_trial_length, 'o')
# plt.legend()

# plt.figure()
# plt.title("Trial length of all trials")
# plt.plot(total_trial_length, 'o')
# plt.legend()

plt.show()
