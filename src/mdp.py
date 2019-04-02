import classifier
import utils
from state import State
import random as rd
import pandas as pd
import numpy as np
import tensorflow as tf

debug_print = False

# class Action:

#   def __init__(self, a):
#       self.action = str(a)

#   def __str__(self):
#       return self.action


class MDP:

    def __init__(self, cluster, model, learning_rate, discount_factor, epsilon, random_state):
        self.random_state = random_state
        self.cluster = cluster
        self.model = model
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        if model in ['ql', 'sarsa']:
            self.policy = dict()
            self.q_table = dict()
        elif model == 'approx':
            pass
        else:
            pass


    def getAction(self, state, randomness):
        actions = state.getLegalActions()
        if rd.random() < randomness:
            return rd.choice(actions)
        candidates = list()
        max_q = float('-inf')
        for a in actions:
            q = self.getQ(state, a)
            if debug_print:
                print('[%s] (%f)' % (a, q))
            if q >= max_q:
                if q > max_q:
                    candidates.clear()
                    max_q = q
                candidates.append(a)
        return rd.choice(candidates)


    def getMaxQ(self, state):
        actions = state.getLegalActions()
        max_q = float('-inf')
        for a in actions:
            max_q = max(max_q, self.getQ(state, a))
        return max_q


    def getQ(self, state, action):
        if self.model in ['ql, sarsa']:
            s_a = state.getHash(action)
            return self.q_table.get(s_a, 0.0)
        elif self.model == 'approx':
            return np.dot(self.weights, state.oneHot(self.label_map))
        else:
            return 0.0


    def applyAction(self, state, action, prediction):
        if action == 'eval':
            state_p = None
        else:
            i = state.next_tree
            pred = state.getPred()
            if action == 'visit':
                if isinstance(prediction, pd.Series):
                    pred[i] = prediction.iloc[i]
                else:
                    pred[i] = prediction
            state_p = State(pred=pred, next_tree=i + 1)
        return state_p


    def train(self, data, num_training, test):
        real = data.iloc[:, -1].reset_index(drop=True)
        results = self.cluster.results(data)
        predictions = pd.concat([results, real], axis=1)

        if self.model == 'approx':
            label_map = dict()
            for label in data.iloc[:, -1]:
                if label not in label_map:
                    label_map[label] = len(label_map)
            for label in test.iloc[:, -1]:
                if label not in label_map:
                    label_map[label] = len(label_map)
            self.label_map = label_map
            self.weights = [0.0] * len(label_map) * self.cluster.size

        n = len(predictions.index)
        for i in range(num_training):
            # shuffle incidents
            shuffled = predictions.sample(frac=1, random_state=self.random_state)
            for j in range(n):
                if self.model == 'ql':
                    self.qLearning(shuffled.iloc[j])
                elif self.model == 'sarsa':
                    self.sarsa(shuffled.iloc[j])
                elif self.model == 'approx':
                    self.approx(shuffled.iloc[j])
            if (i + 1) % 1000 == 0:
                print('Episode %d: accuracy: %f' % (i + 1, self.accuracy(test)))
                # if self.model == 'approx':
                    # print(self.weights)
        if self.model in ['ql', 'sarsa']:
            print(len(self.q_table))


    def qLearning(self, prediction):
        state = State(size=self.cluster.size)
        while state != None:
            if debug_print:
                print('\n{%s}' % (state))
            action = self.getAction(state, self.epsilon)
            state_p = self.applyAction(state, action, prediction)
            # compute factors for updating Q value
            s_a = state.getHash(action)
            q_sa = self.getQ(state, action)
            reward = 0
            if action == 'eval':
                q_sp = 0
                # compute reward
                real = prediction.iloc[-1]
                pred = utils.majorityVote(state)
                if pred == real:
                    reward = 1.0
                else:
                    reward = -1.0
            else:
                # compute max Q for next state
                q_sp = float('-inf')
                actions = state_p.getLegalActions()
                for a in actions:
                    q_sp = max(q_sp, self.getQ(state_p, a))
            self.q_table[s_a] = q_sa + self.learning_rate * (reward + self.discount_factor * q_sp - q_sa)
            if debug_print:
                print('[%s]->{%s}' % (action, str(state_p)))
                print('[%f] [%f] (%f)->(%f)\n' % (reward, q_sp, q_sa, self.q_table[s_a]))
            # update current state
            state = state_p


    def sarsa(self, prediction):
        state = State(size=self.cluster.size)
        action = self.getAction(state, self.epsilon)
        while state != None:
            if debug_print:
                print('\n{%s}' % (state))
            state_p = self.applyAction(state, action, prediction)
            if state_p != None:
                action_p = self.getAction(state_p, self.epsilon)
            else:
                action_p = None
            # compute factors for updating Q value
            if action == 'eval':
                q_sp = 0
                real = prediction.iloc[-1]
                pred = utils.majorityVote(state)
                if pred == real:
                    reward = 1.0
                else:
                    reward = -1.0
            else:
                q_sp = self.getQ(state_p, action_p)
                reward = 0
            s_a = state.getHash(action)
            q_sa = self.getQ(state, action)
            self.q_table[s_a] = q_sa + self.learning_rate * (reward + self.discount_factor * q_sp - q_sa)
            if debug_print:
                print('\n\n{%s}->[%s]->{%s}->[%s]' % (str(state), action, str(state_p), action_p))
                print('[%f] [%f] (%f)->(%f)' % (reward, q_sp, q_sa, self.q_table[s_a]))
            # update current state and action
            state = state_p
            action = action_p


    def approx(self, prediction):
        state = State(size=self.cluster.size)
        action = self.getAction(state, self.epsilon)
        while state != None:
            if debug_print:
                print('\n{%s}' % (state))
            state_p = self.applyAction(state, action, prediction)
            if state_p != None:
                action_p = self.getAction(state_p, self.epsilon)
            else:
                action_p = None
            # compute q, q', r, delta
            if action == 'eval':
                q_sp = 0
                real = prediction.iloc[-1]
                pred = utils.majorityVote(state)
                if pred == real:
                    reward = 1.0
                else:
                    reward = -1.0
            else:
                q_sp = self.getQ(state_p, action_p)
                reward = 0
            q_sa = self.getQ(state, action)
            delta = self.learning_rate * (reward + self.discount_factor * q_sp - q_sa)
            # update weights vector
            one_hot = state.oneHot(self.label_map)
            for i in range(len(one_hot)):
                if one_hot[i] != 0:
                    self.weights[i] += delta / one_hot[i]
            if debug_print:
                print('\n\n{%s}->[%s]->{%s}->[%s]' % (str(state), action, str(state_p), action_p))
                print('[%f] [%f] (%f)->(%f)' % (reward, q_sp, q_sa, self.getQ(state, action)))
            # update current state and action
            state = state_p
            action = action_p


    def featureVote(self, state):
        value = state.getPred()
        vote = dict()
        for i, v in enumerate(value):
            if v is None: continue
            size = len(self.cluster.features[i])
            curr_v = vote.get(v, 0)
            vote[v] = curr_v + size
        pairs = [(v, k) for k, v in vote.items()]
        if len(pairs) > 0:
            _, res = max(pairs)
        else:
            res = None
        return res


    def majorityVote(self, state):
        value = state.getPred()
        vote = dict()
        for v in value:
            if v is None: continue
            curr_v = vote.get(v, 0)
            vote[v] = curr_v + 1
        max_v = 0
        candidates = list()
        for k, v in vote.items():
            if v >= max_v:
                if v > max_v:
                    candidates.clear()
                    max_v = v
                candidates.append(k)
        if len(candidates) > 0:
            return rd.choice(candidates)
        else:
            return None


    def load(self, filename):
        pass


    def save(self, filename):
        pass


    def accuracy(self, test):
        total = test.shape[0]
        correct = 0
        predictions = self.cluster.results(test)
        for i in range(total):
            state = State(size=self.cluster.size)
            # get predicted result
            while state != None:
                action = self.getAction(state, 0)
                # show path
                if debug_print:
                    print('{%s}->[%s]' % (str(state), action))
                if action == 'eval':
                    pred = utils.majorityVote(state)
                    break
                state_p = self.applyAction(state, action, predictions.iloc[i])
                state = state_p
            # get real result & modify counter
            real = test.iloc[i, -1]
            if pred == real:
                correct += 1
            if debug_print:
                print('pred: %s, real: %s, correct: %s\n' % (pred, real, pred == real))
        accuracy = correct / total
        return accuracy



