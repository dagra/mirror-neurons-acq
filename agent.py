"""This file contains the agent."""
from collections import OrderedDict

import numpy as np

from actions import (Eat, GraspJaws, BringToMouth, ReachFood, ReachTube,
                     GraspPaw, Rake, LowerNeck, RaiseNeck, IrrelevantAction)
from acq_parameters import ACQparameters as ACQprms

AVAILABLE_ACTIONS = [Eat, GraspJaws, BringToMouth, ReachFood, ReachTube,
                     GraspPaw, Rake, LowerNeck, RaiseNeck, IrrelevantAction]


class Agent:
    def __init__(self, use_mirror_system=False, n_irrelevant_actions=100,
                 hunger=1, v_max=35):
        self.v_max = v_max
        self.ex_max = 1
        self.ex_min = -0.5
        self.use_mirror_system = use_mirror_system
        self.learn = True

        self.n_irr_actions = n_irrelevant_actions
        self.actions = [a() for a in AVAILABLE_ACTIONS[:-1]]
        self.actions = OrderedDict()
        for i in range(len(AVAILABLE_ACTIONS[:-1])):
            self.actions[AVAILABLE_ACTIONS[i]().name] = AVAILABLE_ACTIONS[i]()
        print "********"
        print map(lambda x: x.name, self.actions.values())
        print "********"
        self.n_rel_actions = len(self.actions.values())

        for i in range(n_irrelevant_actions):
            self.actions[AVAILABLE_ACTIONS[-1]().name + str(i)] = \
                AVAILABLE_ACTIONS[-1]()

        self.hunger = hunger

        self.n_actions = len(self.actions.values())

        # self.w_is = np.random.rand(self.n_actions) - 0.5
        # Initialize desirability weights to 0
        self.w_is = np.zeros(self.n_actions)

        # self.w_pf = np.random.rand(v_max, v_max, self.n_actions)
        # self.w_mf = np.random.rand(v_max, v_max, self.n_actions)
        # self.w_bf = np.random.rand(v_max, v_max, self.n_actions)
        # self.w_pb = np.random.rand(v_max, v_max, self.n_actions)
        # Initialize executability weights to 1
        self.w_pf = np.ones((v_max, v_max, self.n_actions)) * self.ex_max
        self.w_mf = np.ones((v_max, v_max, self.n_actions)) * self.ex_max
        self.w_bf = np.ones((v_max, v_max, self.n_actions)) * self.ex_max
        self.w_pb = np.ones((v_max, v_max, self.n_actions)) * self.ex_max

        self.hist_desirability = self.w_is[:self.n_rel_actions][...,
                                                                np.newaxis]

        self.action_counter = np.zeros(self.n_actions)

    def get_internal_state(self):
        return self.hunger

    def get_desirability(self, internal_state, noise):
        # SOS is e_d different for each action?
        return internal_state * self.w_is + int(noise) * \
            np.asarray([ACQprms.e_d() for i in range(self.n_actions)])

    def get_executability(self, percept, noise):
        # SOS is e_e different for each action?
        ex = np.zeros(self.n_actions)
        for i in range(self.n_actions):
            if i >= self.n_rel_actions:
                ex[i] = 1
                continue
            # if i == 0 and not noise:
            #     print np.sum(self.w_pf[..., i])
            #     print np.sum(self.w_pf[..., i] * percept['pf'])
            #     print np.sum(self.w_mf[..., i])
            #     print np.sum(self.w_mf[..., i] * percept['mf'])
            #     print np.sum(self.w_bf[..., i])
            #     print np.sum(self.w_bf[..., i] * percept['bf'])
            #     print np.sum(self.w_pb[..., i])
            #     print np.sum(self.w_pb[..., i] * percept['pb'])
            ex[i] = np.sum(self.w_pf[..., i] * percept['pf'] +
                           self.w_mf[..., i] * percept['mf'] +
                           self.w_bf[..., i] * percept['bf'] +
                           self.w_pb[..., i] * percept['pb'] +
                           noise * ACQprms.e_e())
            # Set max executability 1
            # Set all irrelevant actions as executable
            # if ex[i] > 1:
            #     ex[i] = 1
            # if ex[i] < 0:
            #     ex[i] = 0
        ex[ex > 1] = 1
        ex[ex < 0 ] = 0
        return ex

    def act(self, env):
        self.env = env
        percept = self.perceive(env)
        selected_action, selected_action_i, desir = self.actor(
            percept, self.get_internal_state(), True)
        executable = selected_action.preconditions(env)
        # if selected_action.name != 'irrelevant_action':
        #     print selected_action.name, p
        #     env.print_current_state()
        # env.visualize_population_codes()
        # if executable and \
        #    selected_action.name != 'irrelevant_action':
        if executable:
            r_signal = selected_action.effects(env, self)
        else:
            r_signal = 0
        # env.print_current_state()
        # if selected_action.name != 'irrelevant_action' and executable:
        #     print selected_action.name, executable
        # perceived_action = self.mirror_system(percept)

        if self.learn:
            r_ex = self.get_executability_reinforcement(selected_action,
                                                        executable)

            r_des, eligibility_trace = self.get_desirability_reinforcement(
                selected_action_i, env, executable, r_signal, desir)
            # if r != 0:
            #     print "*******"
            #     print r, np.nonzero(eligibility_trace),
            #     print selected_action.name, selected_action_i, self.w_is[selected_action_i]
            #     print "*******"
            # print selected_action.name
            self.update_executability(r_ex, percept)
            # if selected_action.type != 'irrelevant':
            #     print selected_action.name, executable
            #     print self.get_executability(percept, False)
            #     # for i in range(len(percept.values())):
            #     #     print np.all(percept.values()[i] == self.perceive(env).values()[i])
            #     print "*****"
            self.update_desirability(r_des, eligibility_trace)
        self.hist_desirability = np.hstack(
            (self.hist_desirability, self.w_is[:self.n_rel_actions][...,
                                                                    np.newaxis]
             )
        )
        self.action_counter[selected_action_i] += 1
        # print "----------------"
        # print self.actions['eat'].preconditions(self.env)
        # print "****************"
        return executable

    def get_executability_reinforcement(self, selected_action, executable):
        if self.use_mirror_system:
            return 0
        reinforce = np.asarray(map(lambda x: x == selected_action,
                                   self.actions.values())).astype('int')
        if not executable:
            reinforce *= -1
        else:
            reinforce *= 1
        return reinforce

    def update_executability(self, reinforce, prev_state):
        for i in range(self.n_actions):
            self.w_pf[..., i] += ACQprms.a * reinforce[i] * prev_state['pf']
            self.w_mf[..., i] += ACQprms.a * reinforce[i] * prev_state['mf']
            self.w_bf[..., i] += ACQprms.a * reinforce[i] * prev_state['bf']
            self.w_pb[..., i] += ACQprms.a * reinforce[i] * prev_state['pb']

        # Threshold executability weights between lb and ub (-5 and 1)
        lb, ub = self.ex_min, self.ex_max
        self.w_pf[self.w_pf < lb] = lb
        self.w_mf[self.w_mf < lb] = lb
        self.w_bf[self.w_bf < lb] = lb
        self.w_pb[self.w_pb < lb] = lb

        self.w_pf[self.w_pf > ub] = ub
        self.w_mf[self.w_mf > ub] = ub
        self.w_bf[self.w_bf > ub] = ub
        self.w_pb[self.w_pb > ub] = ub

    def get_desirability_reinforcement(self, selected_action_i, env,
                                       executable, r_signal, desir):
        if not executable or \
           self.actions.values()[selected_action_i].type == 'irrelevant':
            return 0, np.zeros(self.n_actions)

        if self.use_mirror_system:
            return 0
        # reinforce = (self.actions.values() == selected_action).astype('int')
        _, next_action_i, next_des = self.actor(self.perceive(env),
                                                self.get_internal_state(),
                                                False)
        # SOS reinforce may be negative, how to correct
        reinforce = r_signal + ACQprms.gamma * self.w_is[next_action_i] - \
            self.w_is[selected_action_i]
        if self.actions.keys()[selected_action_i] == 'grasp_jaws':
            print self.actions.keys()[selected_action_i]
            print reinforce, r_signal, next_des, self.w_is[next_action_i], desir
            print selected_action_i, next_action_i
            print self.get_executability(self.perceive(env), False)
            env.print_current_state()

        if -1e03 < reinforce < +1e-03:
            reinforce = 0
        # if reinforce != 0:
        #     print r_signal, self.w_is[next_action_i], next_action_i
        eligibility_trace = np.zeros(self.n_actions)
        eligibility_trace[selected_action_i] = 1
        return reinforce, eligibility_trace

    def update_desirability(self, reinforce, eligibility_trace):
        self.w_is += ACQprms.a * reinforce * eligibility_trace

    def perceive(self, env):
        # env.compute_population_codes()
        return env.get_population_codes()

    def actor(self, percept, internal_state, noise):
        # selected_action = np.random.choice(self.actions.values())
        # if self.actions['eat'].preconditions(self.env):
        #     selected_action = self.actions['eat']
        #     selected_action_i = 0
        # elif self.actions['grasp_jaws'].preconditions(self.env):
        #     selected_action = self.actions['grasp_jaws']
        #     selected_action_i = 1
        # elif self.actions['bring_to_mouth'].preconditions(self.env):
        #     selected_action = self.actions['bring_to_mouth']
        #     selected_action_i = 2
        # elif self.actions['grasp_paw'].preconditions(self.env):
        #     selected_action = self.actions['grasp_paw']
        #     selected_action_i = 3
        # elif self.actions['reach_food'].preconditions(self.env):
        #     selected_action = self.actions['reach_food']
        #     selected_action_i = 4
        # elif self.actions['reach_tube'].preconditions(self.env):
        #     selected_action = self.actions['reach_tube']
        #     selected_action_i = 5
        # return selected_action, selected_action_i
        e = self.get_executability(percept, noise)
        e[e > 1] = 1
        e[e < 0 ] = 0
        d = self.get_desirability(internal_state, noise)
        priority = e * d
        max_inds = np.argwhere(priority == np.max(priority)).flatten()
        selected_action_i = np.random.choice(max_inds)
        selected_action = self.actions.values()[selected_action_i]
        return selected_action, selected_action_i, d[selected_action_i]
