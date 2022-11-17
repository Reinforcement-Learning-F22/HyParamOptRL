import itertools
import numpy as np
from progressbar import progressbar
from utils import epsilon_greedy_policy


class ObjModel:
    def __init__(self, base_model, params, data):
        self.data = data
        self.params = params
        self.params_names = self.params.keys()
        self.comb = self.gen_comb()
        self.observation_space_n = len(self.comb)
        self.action_space_n = len(self.comb)
        self.base_model = base_model
        self.model = None
        self.reset()

    def reset(self):
        state = self.goto_state(0)
        return state

    def goto_state(self, state):
        kwargs = {k: v for k, v in zip(self.params_names, self.comb[state])}
        self.model = self.base_model(**kwargs)
        return state

    def get_params_by_state(self, state):
        return {k: v for k, v in zip(self.params_names, self.comb[state])}

    def train(self):
        self.model.fit(self.data[0], self.data[1])

    def reward(self):
        return self.model.score(self.data[2], self.data[3])

    def gen_comb(self):
        comb = list(itertools.product(*[self.params[i] for i in self.params]))
        return comb

    def step(self, action):
        state = self.goto_state(action)
        self.train()
        r = self.reward()
        return state, r


def train(epsilon, decay_rate, env, max_steps, qtable, gamma, learning_rate):
    episode_rewards = [[], []]
    reward_old = 0
    reward_best = 0
    # Reset the environment
    state = env.reset()
    penalty = 10
    action_best = 0

    # repeat
    for step in progressbar(range(max_steps)):
        # Choose the action At using epsilon greedy policy
        action = epsilon_greedy_policy(qtable, state, epsilon * np.exp(-decay_rate * step))

        # Take action At and observe Rt+1 and St+1
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward_ = env.step(action)
        reward = reward_ if reward_ >= reward_old else (reward_ - reward_old) * penalty
        if reward_ >= reward_best:
            action_best = action
            if step > 0:
                reward *= 2
            reward_best = reward_
        reward_old = reward_

        action_ = epsilon_greedy_policy(qtable, new_state, epsilon)

        # Update Q(s,a)
        qtable[state][action] += learning_rate * (
                reward + gamma * qtable[new_state][action_] - qtable[state][action])

        # Our state is the new state
        state = new_state
        episode_rewards[0].append(reward_)
        episode_rewards[1].append(reward)
    return qtable, episode_rewards, action_best
