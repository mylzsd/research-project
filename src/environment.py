from collections import Counter
import random as rd
import numpy as np


class State:

    def __init__(self, size, label_map):
        self.size = size
        self.label_map = label_map
        self.pred = np.full((size,), None, dtype=object)
        self.oh_pred = np.zeros(size * len(label_map), dtype=int)
        self.visited = set()

    def setPred(self, index, val):
        if index not in range(self.size):
            raise ValueError('index out of bound')
        self.visited.add(index)
        if val is not None:
            self.pred[index] = val
            oh_index = index * len(self.label_map) + self.label_map[val]
            self.oh_pred[index] = 1

    def getPred(self, one_hot=False):
        if one_hot:
            return np.copy(np.reshape(self.oh_pred, (1, -1)))
        else:
            return np.copy(self.pred)

    def usedClf(self):
        return self.visited.copy()

    def copy(self):
        ret = State(self.size, self.label_map)
        for i, v in enumerate(self.pred):
            if v is not None:
                ret.setPred(i, v)
        return ret

    def evaluation(self):
        c = Counter([p for p in self.pred if p is not None])
        if (len(c) == 0):
            return rd.choice(list(self.label_map.keys()))
        m = max(c.values())
        return rd.choice([k for (k, v) in c.items() if v == m])

    def __str__(self):
        return ' '.join(str(e) for e in self.pred)


class Action:

    def __init__(self, index, visit=None):
        self.index = index
        self.visit = visit

    def __str__(self):
        if self.index == -1:
            return 'evaluation'
        elif self.visit is None:
            return 'visit ' + str(self.index)
        else:
            return ('visit ' if self.visit else 'skip ') + str(self.index)


class Environment:
    # read in training and test results, including probabilistic results
    # whether act sequentially, label map, and feature related info
    def __init__(self,
                 num_clf,
                 real_set,
                 res_set, 
                 prob_set, 
                 label_map, 
                 sequential=True,
                 features=None, 
                 feature_cost=None):
        self.num_clf = num_clf
        self.sequential = sequential
        self.real_set = real_set
        self.res_set = res_set
        self.prob_set = prob_set
        self.label_map = label_map

    # return an initial blank state
    def initState(self):
        return State(self.num_clf, self.label_map)

    # return number of instance in a dataset
    def numInstance(self, index):
        return self.real_set[index].shape[0]

    # perform state transition s, a -> s', r
    def step(self, state, action, in_set, in_row, deter=True):
        if deter:
            state_p = state.copy()
            reward = 0.0
            if action.index >= 0:
                # get result of target classifier and perform transition
                val = None
                if action.visit:
                    val = self.res_set[in_set].iloc[in_row, action.index]
                state_p.setPred(action.index, val)
            else:
                # evaluation and get reward
                state_p = None
                pred = state.evaluation()
                if pred == self.real_set[in_set].iloc[in_row]:
                    reward = 1.0
                else:
                    reward = -1.0
        else:
            # TODO: nondeterministic state transition using probabilistic predictions
            pass
        return (state_p, reward)

    # get possible actions
    def legal_actions(self, state):
        if state is None:
            return []
        if self.sequential:
            used = sorted(state.usedClf())
            index = used[-1] + 1 if len(used) > 0 else 0
            if index == self.num_clf:
                return [Action(-1)]
            else:
                return [Action(index, visit=True), Action(index, visit=False)]
        else:
            # TODO: all unvisited classifier as well as evaluation
            unused = set(list(range(self.num_clf))) - state.usedClf()
            ret = [Action(index) for index in unused] + [Action(-1)]
            return ret

    # return the confusion matrix of a given model
    def evaluation(self, model, in_set, deter=True):
        if deter:
            conf_matrix = np.zeros((len(self.label_map), len(self.label_map)), dtype=np.int32)
            for in_row in range(len(self.res_set[in_set])):
                state = self.initState()
                while state is not None:
                    action = model.policy(state)
                    if action.index == -1:
                        pred = state.evaluation()
                        real = self.real_set[in_set].iloc[in_row]
                        conf_matrix[self.label_map[real], self.label_map[pred]] += 1
                    state_p, reward = self.step(state, action, in_set, in_row)
                    state = state_p
            return conf_matrix
        else:
            # TODO: nondeterministic state transition using probabilistic predictions
            pass

