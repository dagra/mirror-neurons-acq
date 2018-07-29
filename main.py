"""Main code"""

import matplotlib.pyplot as plt

from agent import Agent
from external_enviroment import ExternalEnviroment

# Initialization
env = ExternalEnviroment()

agent = Agent(n_irrelevant_actions=0)

h = agent.hunger
i_eat = 0
for i in range(2500):
    agent.act(env)
    if agent.hunger < h:
        print "###########"
        print "Ate!"
        print i, i_eat, agent.hunger
        print "###########"
        agent.hunger = 1
        env = ExternalEnviroment()
        i_eat += 1
    if i_eat > 100:
        break
plt.figure()
n_actions = agent.n_actions - agent.n_irr_actions
labels = map(lambda x: x.name, agent.actions.values()[:n_actions])
colors = map(lambda x: x.color, agent.actions.values()[:n_actions])
print labels
for i in range(n_actions):
    plt.plot(agent.hist_desirability[i, :], label=labels[i], color=colors[i],
             marker='*')
plt.legend()
plt.show()
