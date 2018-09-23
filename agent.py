"""This file contains the agent."""
from collections import OrderedDict

import numpy as np
import torch
from torch.autograd import Variable

from actions import (Eat, GraspJaws, BringToMouth, ReachFood, ReachTube,
                     GraspPaw, Rake, LowerNeck, RaiseNeck, IrrelevantAction)
from acq_parameters import ACQparameters as ACQprms


AVAILABLE_ACTIONS = [Eat, GraspJaws, BringToMouth, GraspPaw, ReachFood,
                     ReachTube, Rake, LowerNeck, RaiseNeck, IrrelevantAction]


def calc_mirror_system_input(current_state, next_state, hunger):
    if np.all(np.asarray(current_state.values()) ==
              np.asarray(next_state.values())):
        return np.append(np.zeros(140), (1 - hunger))
    current_state = np.stack(current_state.values())
    next_state = np.stack(next_state.values())
    # print type(current_state), type(next_state), (np.asarray(current_state.values()) == np.asarray(next_state.values()))
    # state = current_state.values()
    # state.extend(next_state.values())
    # prod = np.linalg.multi_dot(state)
    # state = np.asarray(state).sum(axis=0)
    # state = prod
    diff = (next_state - current_state)
    # state = np.append(diff.sum(axis=1), diff.sum(axis=2))
    state = diff.sum(axis=1) + diff.sum(axis=2)
    return np.append(state.flatten(), (1 - hunger))
    # return np.append((next_state - current_state).flatten(), (1 - hunger))



class Agent:
    def __init__(self, use_mirror_system=False, n_irrelevant_actions=100,
                 hunger=1, v_max=35, learn=True, mirror_system=None,
                 useCuda=False, lesion=False):
        self.v_max = v_max
        self.ex_max = 1
        # In the paper ex_min = -5
        self.ex_min = -5
        self.use_mirror_system = use_mirror_system
        self.learn = learn

        self.n_irr_actions = n_irrelevant_actions
        # self.actions = [a(lesion=lesion) for a in AVAILABLE_ACTIONS[:-1]]
        self.actions = OrderedDict()
        for i in range(len(AVAILABLE_ACTIONS[:-1])):
            self.actions[AVAILABLE_ACTIONS[i]().name] = AVAILABLE_ACTIONS[i]()
        self.n_rel_actions = len(self.actions.values())

        for i in range(n_irrelevant_actions):
            self.actions[AVAILABLE_ACTIONS[-1]().name + str(i)] = \
                AVAILABLE_ACTIONS[-1]()

        self.hunger = hunger

        self.n_actions = len(self.actions.values())

        self.mirror_system = mirror_system

        # Initialize desirability weights to 0
        self.w_is = np.zeros(self.n_actions)

        # Initialize executability weights to 1
        self.w_pf = np.ones((v_max, v_max, self.n_actions)) * self.ex_max
        self.w_mf = np.ones((v_max, v_max, self.n_actions)) * self.ex_max
        self.w_bf = np.ones((v_max, v_max, self.n_actions)) * self.ex_max
        self.w_pb = np.ones((v_max, v_max, self.n_actions)) * self.ex_max

        self.hist_desirability = self.w_is[:self.n_rel_actions][...,
                                                                np.newaxis]

        self.action_counter = np.zeros(self.n_actions)
        self.dw_pf = np.zeros((v_max, v_max, self.n_actions))
        self.dw_mf = np.zeros((v_max, v_max, self.n_actions))
        self.dw_bf = np.zeros((v_max, v_max, self.n_actions))
        self.dw_pb = np.zeros((v_max, v_max, self.n_actions))

        self.useCuda = useCuda

    def act(self, env):
        """Perform an action in the enviroment.

        Perceive the enviroment, choose an action based on the actor,
        execute the action (if possible) and learn.

        This is the main method that should be called externally.
        """
        self.env = env
        percept = self.perceive(env)
        self.current_state = percept
        selected_action, selected_action_i, des_vec, priority = self.actor(
            percept, self.get_internal_state(), True)
        self.training_signal = np.asarray(
            map(lambda x: x == selected_action,
                self.actions.values())).astype('float')
        executable = selected_action.preconditions(env)
        if executable:
            r_signal = selected_action.effects(env, self)
            self.next_state = self.perceive(env)
        else:
            r_signal = 0
            self.next_state = self.current_state

        if self.use_mirror_system:
            ms_output = self.get_mirror_system_output(priority)
            action_rec = np.max(ms_output)
            recognized_action_i = np.argwhere(ms_output == action_rec)[0][0]
            recognized_action = self.actions.values()[selected_action_i]
        else:
            recognized_action_i = selected_action_i
            recognized_action = selected_action
            ms_output = None

        if self.learn:
            r_ex = self.get_executability_reinforcement(recognized_action,
                                                        executable, priority,
                                                        ms_output)

            desir = des_vec[recognized_action_i]
            r_des, eligibility_trace = self.get_desirability_reinforcement(
                recognized_action_i, env, executable, r_signal, desir, ms_output)

            self.update_executability(r_ex, percept)

            self.update_desirability(r_des, eligibility_trace)

        # Record some stats
        self.hist_desirability = np.hstack(
            (self.hist_desirability, self.w_is[:self.n_rel_actions][...,
                                                                    np.newaxis]
             )
        )
        if executable:
            self.action_counter[selected_action_i] += 1
        return executable

    def actor(self, percept, internal_state, noise):
        """Choose and action based on the enviroment and the internal state"""
        # This is the main strategy that should be learned, as described
        # in the paper

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
        d = self.get_desirability(internal_state, noise)
        priority = e * d
        max_inds = np.argwhere(priority == np.max(priority)).flatten()
        selected_action_i = np.random.choice(max_inds)
        selected_action = self.actions.values()[selected_action_i]
        return selected_action, selected_action_i, d,\
            priority

    def get_executability_reinforcement(self, selected_action, executable,
                                        priority, ms_output):
        """Compute executability reinforcement"""
        if self.use_mirror_system:
            reinforce = np.zeros(self.n_actions)
            max_priority = np.max(priority)
            for i in range(len(ms_output)):
                # x is ambiguous in the paper i.e. is it the priority vector
                # or an one-hot vector containing the selected action?
                # if priority[i] > 0 and ms_output[i] < ACQprms.psi:
                #     reinforce[i] = -1
                if priority[i] == max_priority and \
                   ms_output[i] < ACQprms.psi:
                    reinforce[i] = -1
                elif ms_output[i] > ACQprms.psi:
                    reinforce[i] = 1
            if executable and selected_action.name in ('grasp_paw'):
                # print priority, ms_output, reinforce, selected_action.name, executable, np.all(ms_input == 0)
                print ms_output, reinforce, selected_action.name, executable
                self.env.print_current_state()
            return reinforce.astype('float')
        # If the mirror system is absent, the paper doesn't describe what
        # the reinforcement is.
        # So the assumption here is that only the selected action is
        # influenced according to the executability.
        reinforce = np.asarray(map(lambda x: x == selected_action,
                                   self.actions.values())).astype('float')
        if not executable:
            reinforce *= -1
        else:
            reinforce *= 1
        return reinforce

    def update_executability(self, reinforce, prev_state, momentum=0):
        """Update executability weights.

        For every action, compute the change based on the reinforcement
        signal and add it to the weights.

        An additional, optional parameter (momentum) is added that doesn't
        exist in the paper's rule. For momentum=0 the two rules are equal.
        """
        for i in range(self.n_actions):
            self.dw_pf[..., i] = self.dw_pf[..., i] * momentum + \
                ACQprms.a * reinforce[i] * prev_state['pf'] * (1 - momentum)
            self.dw_mf[..., i] = self.dw_mf[..., i] * momentum + \
                ACQprms.a * reinforce[i] * prev_state['mf'] * (1 - momentum)
            self.dw_bf[..., i] = self.dw_bf[..., i] * momentum + \
                ACQprms.a * reinforce[i] * prev_state['bf'] * (1 - momentum)
            self.dw_pb[..., i] = self.dw_pb[..., i] * momentum + \
                ACQprms.a * reinforce[i] * prev_state['pb'] * (1 - momentum)
            self.w_pf[..., i] += self.dw_pf[..., i]
            self.w_mf[..., i] += self.dw_mf[..., i]
            self.w_bf[..., i] += self.dw_bf[..., i]
            self.w_pb[..., i] += self.dw_pb[..., i]
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
                                       executable, r_signal, desir,
                                       ms_output):
        """Compute desirability reinforcement"""
        if not self.use_mirror_system and (not executable or \
           self.actions.values()[selected_action_i].type == 'irrelevant'):
            return 0, np.zeros(self.n_actions)
        if self.actions.values()[selected_action_i].type == 'irrelevant':
            return 0, np.zeros(self.n_actions)

        # Compute next selected action, but without noise
        _, next_action_i, next_des_vec, p = self.actor(
            self.perceive(env),
            self.get_internal_state(),
            False)
        next_des = next_des_vec[next_action_i]

        if self.use_mirror_system:
            reinforce = r_signal + ACQprms.gamma * next_des - desir
            reinforce = r_signal + ACQprms.gamma * next_des - \
                self.w_is[selected_action_i]
            eligibility_trace = np.zeros(self.n_actions)
            eligibility_trace[:self.n_rel_actions] = \
                (ms_output > ACQprms.s_p).astype('int')
            # eligibility_trace[:self.n_rel_actions] = \
            #     (ms_output > 0).astype('int')
            # eligibility_trace = np.tanh(ms_output)
            if reinforce > 0 and self.w_is[selected_action_i] > 0.95:
                print self.actions.keys()[selected_action_i]
                print "r={}, signal={}, next_des={}, weight={}, desir={}".format(
                    reinforce, r_signal, next_des, self.w_is[selected_action_i], desir)
                print "selected action={}, next={}".format(selected_action_i,
                                                           next_action_i)
                print "next_weight={}".format(self.w_is[next_action_i])
                print self.get_executability(self.perceive(env), False)
                env.print_current_state()
            return reinforce, eligibility_trace

        # If the mirror system is absent, the paper doesn't describe what
        # the reinforcement is.
        # Also it isn't explicitly described how the expected desirability of
        # the next step is computed.
        # The assumptions here are:
        # 1. The desirability of the next step corresponds to the desirability
        #    of the next selected action using the current policy
        # 2. The eligibility trace is only for the selected action

        # As desirability either the weight or the noised(-free?) computed
        # desirability (next_des) can be used
        # Both alternatives are written below.
        # Because noise is inside the given desirability of the selected action
        # these alternatives are NOT equal.
        reinforce = r_signal + ACQprms.gamma * next_des - \
            self.w_is[selected_action_i]

        # Debug
        # if self.actions.keys()[selected_action_i] in ()
           
        #     print self.actions.keys()[selected_action_i]
        #     print "r={}, signal={}, next_des={}, weight={}, desir={}".format(
        #         reinforce, r_signal, next_des, self.w_is[next_action_i], desir)
        #     print "selected action={}, next={}".format(selected_action_i,
        #                                                next_action_i)
        #     print self.get_executability(self.perceive(env), False)
        #     env.print_current_state()

        # if -1e018 < reinforce < +1e-018:
        #     reinforce = 0
        # Compute eligibility trace
        eligibility_trace = np.zeros(self.n_actions)
        eligibility_trace[selected_action_i] = 1
        return reinforce, eligibility_trace

    def update_desirability(self, reinforce, eligibility_trace):
        self.w_is += ACQprms.a * reinforce * eligibility_trace

    def get_mirror_system_output(self, x):
        ms_input = calc_mirror_system_input(self.current_state,
                                            self.next_state, self.hunger)
        FloatTensor = torch.cuda.FloatTensor if self.useCuda \
            else torch.FloatTensor
        out = self.mirror_system(Variable(torch.from_numpy(
            np.asarray([ms_input])).type(FloatTensor)))
        if self.useCuda:
            out = out.cpu().data.numpy()[0]
        else:
            out = out.data.numpy()[0]
        out += ACQprms.k * x[:self.n_rel_actions]
        _max = np.max(np.abs(out))
        return out / _max

    def get_internal_state(self):
        return self.hunger

    def get_desirability(self, internal_state, noise):
        return internal_state * self.w_is + int(noise) * \
            np.asarray([ACQprms.e_d() for i in range(self.n_actions)])

    def get_executability(self, percept, noise):
        ex = np.zeros(self.n_actions)
        for i in range(self.n_actions):
            if i >= self.n_rel_actions:
                ex[i] = 1
                continue
            ex[i] = np.sum(self.w_pf[..., i] * percept['pf'] +
                           self.w_mf[..., i] * percept['mf'] +
                           self.w_bf[..., i] * percept['bf'] +
                           self.w_pb[..., i] * percept['pb']) +\
                noise * ACQprms.e_e()
        # Set max executability 1
        # Set all irrelevant actions as executable
        ex[ex > 1] = 1
        ex[ex < 0] = 0
        return ex

    def perceive(self, env):
        # env.compute_population_codes()
        return env.get_population_codes()

    def apply_lesion(self):
        for a in self.actions.values():
            a.lesion = True
