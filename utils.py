import numpy as np
import torch
from dataclasses import make_dataclass
from collections import Counter
from sklearn.metrics import accuracy_score


# =========== Reward utils ===========
def get_best_params(qtable, env, calc_acc=False):
    coef = [0.1, 0.3]

    x = [np.argmax(qtable[i]) for i in range(len(qtable))]
    y = [
        [rec_state(x, i, int(len(qtable)*coef[0])) for i in range(len(qtable))],
        [rec_state(x, i, int(len(qtable)*coef[1])) for i in range(len(qtable))]
    ]
    states = [Counter(i).most_common()[0][0] for i in y]
    if calc_acc:
        accs = [env.step(i)[1] for i in states]
        return [[{k: v for k, v in zip(env.params_names, env.comb[state])}, acc, state] for state, acc in zip(states, accs)]
    else:
        return [[{k: v for k, v in zip(env.params_names, env.comb[state])}, state] for state in states]


# =========== RL utils ===========
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


# =========== Dataset utils ===========
def gen_dataclass(kwargs_, name=''):
    dataclass_ = make_dataclass(name, kwargs_.keys())
    return dataclass_(*kwargs_.values())


def convert2torch(*arrs):
    result = [torch.from_numpy(x).float() for x in arrs]
    return result


def generate_features_values(prefix, size, index=1):
    a = np.arange(index, size + index)
    return [prefix + str(i) for i in a]


def train_test_split(X, y, S, test_size=0.3):
    split_size = int(X.shape[0] * test_size)
    X_test, y_test, s_test = X[0:split_size, :], y[0:split_size], S[0:split_size]
    X_train, y_train, s_train = X[split_size + 1:, :], y[split_size + 1:], S[split_size + 1:]
    return X_train, X_test, y_train, y_test, s_train, s_test


# =========== Fairness metrics ===========
def confusion_matrix_score(y_pred, y_true, s):
    """
        Parameters
        ----------
        y_pred : 1-D array size n
            Label returned by the model
        y_true : 1-D array size n
            Real label
            # print("Training %s"%(name))
        s: 1-D size n protected attribut
        Return
        -------
        equal_opportunity True positive error rate across group
        equal_disadvantage False positive error rate across group
    """

    alpha_1 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == 1, s == 0))) / float(
        np.sum(np.logical_and(y_true == 1, s == 0)))
    beta_1 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == 1, s == 1))) / float(np.sum(
        np.logical_and(y_true == 1, s == 1)))

    alpha_2 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == 0, s == 0))) / float(np.sum(
        np.logical_and(y_true == 0, s == 0)))
    beta_2 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == 0, s == 1))) / float(np.sum(
        np.logical_and(y_true == 0, s == 1)))

    equal_opportunity = np.abs(alpha_1 - beta_1)
    equal_disadvantage = np.abs(alpha_2 - beta_2)
    return equal_opportunity, equal_disadvantage


def cross_val_fair_scores(model, X, y, cv, protected_attrib, fit_sensitive=False):
    """
    model : class with fit and predict methods
    X: features matrices
    y: labels
    cv: Kfold cross validation from Sklearn
    protected_attrib: Protected attribute
    scoring : "statistical_parity_score" | "equalized_odds" | "equal_opportunity"
    fit_sensitive: True if the fit method receive sensitive attribute. Only for fairness-aware estimators
    """
    st_scores = []
    equal_odds = []
    equal_opps = []
    accuracy = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        s_train, s_test = protected_attrib[train_index], protected_attrib[test_index]
        if fit_sensitive:
            clf = model.fit(X_train, y_train, sensitive_features=s_train)
            y_pred = clf.predict(X_test, sensitive_features=s_test)
        else:
            clf = model.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        st_score = statistical_parity_score(y_pred, s_test)
        st_scores.append(st_score)

        tpr, fpr = confusion_matrix_score(y_pred, y_test, s_test)
        equal_odds.append(tpr + fpr)

        tpr, _ = confusion_matrix_score(y_pred, y_test, s_test)
        equal_opps.append(tpr)

        accuracy.append(accuracy_score(y_test, y_pred))
    return accuracy, st_scores, equal_odds, equal_opps


def statistical_parity_score(y_pred, s):
    """ This measure the proportion of positive and negative class in protected and non-protected group """
    alpha_1 = np.sum(np.logical_and(y_pred == 1, s == 1)) / float(np.sum(s == 1))
    beta_1 = np.sum(np.logical_and(y_pred == 1, s == 0)) / float(np.sum(s == 0))
    return np.abs(alpha_1 - beta_1)
