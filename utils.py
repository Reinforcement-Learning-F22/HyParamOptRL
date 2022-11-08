import numpy as np
from collections import Counter


# Let's create our Qtable of size (state_space, action_space) and initialized each values at 0 using np.zeros
def initialize_q_table(state_space, action_space, range_=(-1, 1), zeros=False):
    func = np.zeros if zeros else np.random.random
    qtable = (not zeros) * range_[0] + (range_[1] - range_[0]) * func((state_space, action_space))
    return qtable


def epsilon_greedy_policy(qtable, state, epsilon):
    # Randomly generate a number between 0 and 1
    random_num = np.random.random()
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = np.argmax(qtable[state])
    # else --> exploration
    else:
        action = np.random.choice(np.arange(len(qtable[0])))  # Take a random action
    return action


def rec_state_path(x, i, depth, path_):
    if i > 1000:
        i = 1000
    if i in path_:
        return
    path_.append(i)
    if depth == 0:
        return x[i]
    return rec_state_path(x, x[i], depth - 1, path_)


def rec_state(x, i, depth):
    if i > 1000:
        i = 1000
    if depth == 0:
        return x[i]
    return rec_state(x, x[i], depth - 1)


def get_best_params(qtable, env, calc_acc=False):
    x = [np.argmax(qtable[i]) for i in range(len(qtable))]
    y = [
        [rec_state(x, i, 2) for i in range(len(qtable))],
        [rec_state(x, i, 3) for i in range(len(qtable))]
    ]

    y1 = np.zeros(len(qtable[0]))
    for i in range(len(qtable[0])):
        path = []
        rec_state_path(x, i, len(qtable[0]), path)
        for p in path:
            y1[p] += 1
    states = [Counter(i).most_common()[0][0] for i in y] + [np.argmax(y1)]
    if calc_acc:
        accs = [env.step(i)[1] for i in states]
        return [[{k: v for k, v in zip(env.params_names, env.comb[state])}, acc] for state, acc in zip(states, accs)]
    else:
        return [{k: v for k, v in zip(env.params_names, env.comb[state])} for state in states]

