"""Main code"""

import time

import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
from external_enviroment import ExternalEnviroment

# Initialization
env = ExternalEnviroment()

agent = Agent(n_irrelevant_actions=5)

max_eat = 100
i_eat = 0
executability_error = np.zeros(max_eat)
n_actions = agent.n_actions - agent.n_irr_actions
last_action_desirability = np.zeros((n_actions, max_eat))
while i_eat < max_eat:
    n_tried_actions = 0
    start_time = time.time()
    while n_tried_actions < 50:
        executed = agent.act(env)
        executability_error[i_eat] += int(not executed)
        n_tried_actions += 1
        if agent.hunger == 0:
            print "###########"
            print "{}/{}-actions:{}-time:{} sec".format(i_eat + 1, max_eat,
                                                        n_tried_actions,
                                                        int(time.time() -
                                                            start_time))
            break
    last_action_desirability[:, i_eat] = agent.hist_desirability[:n_actions,
                                                                 -1]
    executability_error[i_eat] /= float(n_tried_actions)
    i_eat += 1
    agent.hunger = 1
    env = ExternalEnviroment()

plt.figure()
n_actions = agent.n_actions - agent.n_irr_actions
labels = map(lambda x: x.name, agent.actions.values()[:n_actions])
colors = map(lambda x: x.color, agent.actions.values()[:n_actions])
print labels
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
