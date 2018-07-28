"""Main code"""

import matplotlib.pyplot as plt

from agent import Agent
from external_enviroment import ExternalEnviroment

# Initialization
env = ExternalEnviroment()

agent = Agent(n_irrelevant_actions=0)

h = agent.hunger
i_eat = 0
for i in range(500):
    agent.act(env)
    if agent.hunger < h:
        print "Eat!"
        print i
        h = agent.hunger
        env = ExternalEnviroment()
        i_eat += 1
    if i_eat > 100:
        break
print i_eat
plt.figure()
n_actions = agent.n_actions - agent.n_irr_actions
labels = map(lambda x: x.name, agent.actions.values()[:n_actions])
print labels
for i in range(n_actions):
    print type(agent.hist_desirability)
    print agent.hist_desirability.shape
    plt.plot(agent.hist_desirability[i, :], label=labels[i])
plt.legend()
plt.show()
