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
                 hunger=100, v_max=35):
        self.v_max = v_max
        self.use_mirror_system = use_mirror_system
        self.learn = True

        self.n_irr_actions = n_irrelevant_actions
        self.actions = [a() for a in AVAILABLE_ACTIONS[:-1]]
        self.actions = OrderedDict()
        for i in range(len(AVAILABLE_ACTIONS)):
            self.actions[AVAILABLE_ACTIONS[i]().name] = AVAILABLE_ACTIONS[i]()
        self.n_rel_actions = len(self.actions.values())
        # self.actions.extend([AVAILABLE_ACTIONS[-1]()
        #                     for i in range(n_irrelevant_actions)])
        self.hunger = hunger

        self.n_actions = len(self.actions.values())
        self.w_is = np.random.rand(self.n_actions) - 0.5
        # Initialize desirability weights to 0
        self.w_is = np.zeros(self.n_actions)

        # self.w_pf = np.random.rand(v_max, v_max, self.n_actions)
        # self.w_mf = np.random.rand(v_max, v_max, self.n_actions)
        # self.w_bf = np.random.rand(v_max, v_max, self.n_actions)
        # self.w_pb = np.random.rand(v_max, v_max, self.n_actions)
        # Initialize executability weights to 1
        self.w_pf = np.ones((v_max, v_max, self.n_actions))
        self.w_mf = np.ones((v_max, v_max, self.n_actions))
        self.w_bf = np.ones((v_max, v_max, self.n_actions))
        self.w_pb = np.ones((v_max, v_max, self.n_actions))

        self.hist_desirability = self.w_is[:self.n_rel_actions][...,
                                                                np.newaxis]

    def get_internal_state(self):
        return self.hunger

    def get_desirability(self, internal_state):
        # SOS is e_d different for each action?
        return internal_state * self.w_is + [ACQprms.e_d() for i in
                                             range(self.n_actions)]

    def get_executability(self, percept):
        # SOS is e_e different for each action?
        ex = np.zeros(self.n_actions)
        for i in range(self.n_actions):
            ex[i] = np.sum(self.w_pf[..., i] * percept['pf'] +
                           self.w_mf[..., i] * percept['mf'] +
                           self.w_bf[..., i] * percept['bf'] +
                           self.w_pb[..., i] * percept['pb'] +
                           ACQprms.e_e())
            # Set max executability 1
            # Set all irrelevant actions as executable
            if ex[i] > 1 or i >= self.n_rel_actions:
                ex[i] = 1
        return ex

    def act(self, env):
        self.env = env
        percept = self.perceive(env)
        selected_action, selected_action_i = self.actor(
            percept,
            self.get_internal_state())
        success = selected_action.preconditions(env)
        # if selected_action.name != 'irrelevant_action':
        #     print selected_action.name, p
        #     env.print_current_state()
        # if success and \
        #    selected_action.name != 'irrelevant_action':
        r_signal = selected_action.effects(env, self)
        # env.print_current_state()
        print selected_action.name
        # perceived_action = self.mirror_system(percept)

        if self.learn:
            r = self.get_executability_reinforcement(selected_action, success)
            self.update_executability(r, percept)
            r, eligibility_trace = self.get_desirability_reinforcement(
                selected_action_i, env, success, r_signal)
            self.update_desirability(r, eligibility_trace)
        self.hist_desirability = np.hstack(
            (self.hist_desirability, self.w_is[:self.n_rel_actions][...,
                                                                    np.newaxis]
             )
        )

    def get_executability_reinforcement(self, selected_action, success):
        if self.use_mirror_system:
            return 0
        reinforce = np.asarray(map(lambda x: x == selected_action,
                                   self.actions.values())).astype('int')
        if not success:
            reinforce *= -1
        return reinforce

    def update_executability(self, reinforce, prev_state):
        for i in range(self.n_actions):
            self.w_pf[..., i] += ACQprms.a * reinforce[i] * prev_state['pf']
            self.w_mf[..., i] += ACQprms.a * reinforce[i] * prev_state['mf']
            self.w_bf[..., i] += ACQprms.a * reinforce[i] * prev_state['bf']
            self.w_pb[..., i] += ACQprms.a * reinforce[i] * prev_state['pb']

        # Threshold executability weights between -5 and 1
        self.w_pf[self.w_pf < -5.0] = -5
        self.w_mf[self.w_mf < -5.0] = -5
        self.w_bf[self.w_bf < -5.0] = -5
        self.w_pb[self.w_pb < -5.0] = -5

        self.w_pf[self.w_pf > 1] = 1
        self.w_mf[self.w_mf > 1] = 1
        self.w_bf[self.w_bf > 1] = 1
        self.w_pb[self.w_pb > 1] = 1

    def get_desirability_reinforcement(self, selected_action_i, env, success,
                                       r_signal):
        if self.use_mirror_system:
            return 0
        # reinforce = (self.actions.values() == selected_action).astype('int')
        _, next_action_i = self.actor(self.perceive(env),
                                      self.get_internal_state())
        reinforce = r_signal + ACQprms.gamma * self.w_is[next_action_i] - \
            self.w_is[selected_action_i]
        eligibility_trace = np.zeros(self.n_actions)
        eligibility_trace[selected_action_i] = 1
        return reinforce, eligibility_trace

    def update_desirability(self, reinforce, eligibility_trace):
        self.w_is += ACQprms.a * reinforce * eligibility_trace

    def perceive(self, env):
        return env.get_population_codes()

    def actor(self, percept, internal_state):
        # selected_action = np.random.choice(self.actions.values())
        # if self.actions['eat'].preconditions(self.env):
        #     selected_action = self.actions['eat']
        # elif self.actions['grasp_jaws'].preconditions(self.env):
        #     selected_action = self.actions['grasp_jaws']
        # elif self.actions['bring_to_mouth'].preconditions(self.env):
        #     selected_action = self.actions['bring_to_mouth']
        # elif self.actions['grasp_paw'].preconditions(self.env):
        #     selected_action = self.actions['grasp_paw']
        # elif self.actions['reach_food'].preconditions(self.env):
        #     selected_action = self.actions['reach_food']
        # elif self.actions['reach_tube'].preconditions(self.env):
        #     selected_action = self.actions['reach_tube']
        e = self.get_executability(percept)
        d = self.get_desirability(internal_state)
        priority = e * d
        max_inds = np.argwhere(priority == np.max(priority)).flatten()
        selected_action_i = np.random.choice(max_inds)
        selected_action = self.actions.values()[selected_action_i]
        return selected_action, selected_action_i
