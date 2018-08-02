"""Main code"""

import time

import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
from external_enviroment import ExternalEnviroment

# Initialization
env = ExternalEnviroment()

agent = Agent(n_irrelevant_actions=0)

max_eat = 100
trial = 0
trial_success = 0
executability_error = np.zeros(max_eat) - 1
n_actions = agent.n_actions - agent.n_irr_actions
last_action_desirability = np.zeros((n_actions, max_eat)) - 1
# while trial < max_eat:
while trial_success < 100:
    n_tried_actions = 0
    start_time = time.time()
    if trial >= len(executability_error):
        executability_error = np.append(executability_error,
                                        np.zeros(max_eat) - 1)
        last_action_desirability = np.append(last_action_desirability,
                                             np.zeros((n_actions,
                                                       max_eat)) - 1,
                                             axis=1)
    while n_tried_actions < 50:
        executed = agent.act(env)
        executability_error[trial] += int(not executed)
        n_tried_actions += 1
        if agent.hunger == 0:
            trial_success += 1
            break
    last_action_desirability[:, trial] = agent.hist_desirability[:n_actions,
                                                                 -1]
    executability_error[trial] /= float(n_tried_actions)
    print "###########"
    print "{}/{}/{}-Ate: {}-actions:{}-exec rate: {}-time:{} sec".format(
        trial_success + 1, trial + 1, max_eat, agent.hunger == 0,
        n_tried_actions, executability_error[trial],
        np.round(time.time() -
                 start_time))
    print agent.action_counter
    agent.action_counter = np.zeros(agent.n_actions)
    trial += 1
    agent.hunger = 1
    env = ExternalEnviroment()

plt.figure()
n_actions = agent.n_actions - agent.n_irr_actions
labels = map(lambda x: x.name, agent.actions.values()[:n_actions])
colors = map(lambda x: x.color, agent.actions.values()[:n_actions])
for i in range(n_actions):
    plt.plot(agent.hist_desirability[i, :], label=labels[i], color=colors[i],
             marker='*')
plt.legend()

plt.figure()
for i in range(n_actions):
    plt.plot(last_action_desirability[i, :], label=labels[i], color=colors[i],
             marker='*')
plt.legend()

plt.figure()
plt.plot(executability_error, marker='*', label='Mean executability error')
plt.legend()

plt.show()
