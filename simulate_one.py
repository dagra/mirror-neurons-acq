"""Main code

Initializes the agent and the enviroment.
Simulates N trials.
Records and plots the corresponding statistics.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agent
from external_enviroment import ExternalEnviroment
from create_mirror_system import (Model, create_network,)


def simulate(agent, verbose=0, plot=0, **kwargs):
    max_eat = kwargs['max_eat']
    max_actions = kwargs['max_actions']
    trial_success = kwargs['trial_success']
    recovery_time = kwargs.get('recovery_time')
    if recovery_time:
        recovery_flag = 0
    lesion_time = kwargs.get('lesion_time')
    i_all_last = kwargs['i_all_last']
    all_last_action_desirability = kwargs['all_last_action_desirability']
    last_action_desirability = kwargs['last_action_desirability']
    success_trial_length = kwargs['success_trial_length']
    total_trial_length = kwargs['total_trial_length']
    success_trial_length = kwargs['success_trial_length']
    trial = kwargs['trial']

    n_rel_actions = agent.n_actions - agent.n_irr_actions
    env = ExternalEnviroment()
    try:
        while trial_success < max_eat and trial < 700:
            n_tried_actions = 0
            while n_tried_actions < max_actions:
                agent.act(env)
                n_tried_actions += 1
                if agent.hunger == 0:
                    last_action_desirability[:, trial_success] = \
                        agent.hist_desirability[:n_rel_actions, -1]
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

            total_trial_length.append(n_tried_actions)
            # Check recovery
            if recovery_time and agent.hunger == 0 and \
               agent.action_counter[6] >= 1 and \
               agent.action_counter[7] >= 1 and \
               agent.w_is[6] > 0.05 and \
               recovery_time[1] == 0:
                recovery_flag += 1
                recovery_time[0] += 1
                if recovery_flag == 3:
                    recovery_time[1] = trial_success - 1
                    recovery_time[2] = i_all_last - 1
            elif recovery_time and recovery_time[1] == 0:
                recovery_flag = 0
                recovery_time[0] += 1

            if verbose:
                print "###########"
                print "[{}]: {}/{}-Ate: {}-actions:{}".format(
                    trial + 1, trial_success, max_eat, agent.hunger == 0,
                    n_tried_actions)
                # Debug
                print agent.action_counter[:n_rel_actions]

            agent.action_counter = np.zeros(agent.n_actions)
            trial += 1
            # Reset agent and enviroment
            agent.hunger = 1
            env.reset()
    except KeyboardInterrupt:
        pass

    if verbose and plot:
        # Plot results
        labels = map(lambda x: x.name, agent.actions.values()[:n_rel_actions])
        colors = map(lambda x: x.color, agent.actions.values()[:n_rel_actions])
        marks = map(lambda x: x.marker, agent.actions.values()[:n_rel_actions])

        plt.figure()
        plt.title("Desirability per action in the last step of all trials")
        for i in range(n_rel_actions):
            plt.plot(all_last_action_desirability[i, :i_all_last],
                     label=labels[i], color=colors[i],
                     marker=marks[i], mfc='none')
        if recovery_time:
            plt.axvline(x=recovery_time[2], color='black', ls='--',
                        label='Recovery')
        if lesion_time:
            plt.axvline(x=lesion_time[1], color='black', label='Lesion')
        plt.legend()

        plt.figure()
        plt.title("Desirability per action in the last step "
                  "of the successful trials")
        for i in range(n_rel_actions):
            plt.plot(last_action_desirability[i, :], label=labels[i],
                     color=colors[i], marker=marks[i], mfc='none')
        if recovery_time:
            plt.axvline(x=recovery_time[1], color='black', ls='--',
                        label='Recovery')
        if lesion_time:
            plt.axvline(x=lesion_time[0], color='black', label='Lesion')
        plt.legend()

        plt.figure()
        plt.title("Trial length of successful trials")
        plt.plot(success_trial_length, 'o')
        plt.legend()

        plt.figure()
        plt.title("Trial length of all trials")
        plt.plot(total_trial_length, 'o')
        plt.legend()

        plt.show()

    output = ['trial_success', 'i_all_last', 'all_last_action_desirability',
              'last_action_desirability', 'success_trial_length',
              'total_trial_length', 'success_trial_length', 'trial']
    if recovery_time:
        output.append('recovery_time')
    results = {}
    for v in output:
        results[v] = eval(v)
    return results


def simulate_one(max_eat, max_actions, use_mirror_system,
                 n_irrelevant_actions, mirror_system=None,
                 useCuda=False, verbose=1):
    # Initialization
    agent = Agent(use_mirror_system=use_mirror_system,
                  n_irrelevant_actions=n_irrelevant_actions,
                  mirror_system=mirror_system, useCuda=useCuda)
    kwargs = {}
    kwargs['max_eat'] = max_eat
    max_eat = max_eat * 2
    kwargs['max_actions'] = max_actions
    # Statistics
    n_rel_actions = agent.n_actions - agent.n_irr_actions
    kwargs['executability_error'] = np.zeros(max_eat) - 1
    kwargs['success_trial_length'] = np.zeros(max_eat) - 1
    kwargs['total_trial_length'] = []
    kwargs['total_executability_error'] = []
    kwargs['last_action_desirability'] = np.zeros((n_rel_actions,
                                                   max_eat)) - 1
    kwargs['all_last_action_desirability'] = np.zeros((n_rel_actions,
                                                       max_eat)) - 1
    kwargs['i_all_last'] = 0
    kwargs['trial'] = 0
    kwargs['trial_success'] = 0

    # Train Agent without the lesion
    output = simulate(agent, verbose=verbose, plot=0, **kwargs)

    agent.apply_lesion()
    output['recovery_time'] = [0, 0, 0]
    output['lesion_time'] = [output['trial_success'], output['i_all_last']]
    output['max_eat'] = max_eat
    output['max_actions'] = max_actions

    # Test Agent with the lesion
    output = simulate(agent, verbose=verbose, plot=verbose, **output)
    # Return number of trials needed for recovery
    return output['recovery_time'][0]


if __name__ == '__main__':
    # Simulation parameters
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

    max_eat = 100
    max_actions = 50
    n_irrelevant_actions = 10

    recovery_time = simulate_one(max_eat, max_actions, use_mirror_system,
                                 n_irrelevant_actions, mirror_system=net,
                                 useCuda=useCuda, verbose=1)

    print "Trials needed for recovery: %d" % recovery_time
