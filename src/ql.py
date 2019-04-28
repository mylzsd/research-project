from collections import Counter
import random as rd


class Tabular:

    def __init__(self, env):
        self.env = env
        self.q_table = Counter()

    def policy(self, state, in_set, in_row, randomness=0.0):
        actions = self.env.legal_actions(state)
        if len(actions) == 0:
            return None
        # exploration or exploitation
        if rd.random() < randomness:
            return rd.choice(actions)
        maxq = float('-inf')
        candidates = []
        for a in actions:
            q = self.q_table[str(state) + str(a)]
            if q >= maxq:
                if q > maxq:
                    maxq = q
                    candidates.clear()
                candidates.append(a)
        return rd.choice(candidates)

    def qValue(self, state, action=None):
        if state is None:
            return 0.0
        if action is not None:
            return self.q_table[str(state) + str(action)]
        actions = self.env.legal_actions(state)
        maxq = float('-inf')
        for a in actions:
            k = str(state) + str(a)
            maxq = max(maxq, self.q_table[k])
        return maxq

    def train(self, state, action, state_p, reward, learning_rate, discount_factor, action_p=None):
        k = str(state) + str(action)
        self.q_table[k] += learning_rate * (reward + discount_factor * self.qValue(state_p, action_p) - self.q_table[k])


def learn(env, in_set, num_training, learning_rate, epsilon, discount_factor, random_state):
    model = Tabular(env)
    num_ins = env.numInstance(in_set)
    for i in range(num_training):
        in_row = i % num_ins
        state = env.initState()
        while state is not None:
            action = model.policy(state, in_set, in_row, randomness=epsilon)
            state_p, reward = env.step(state, action, in_set, in_row)
            # maybe modify model every k steps
            model.train(state, action, state_p, reward, learning_rate, discount_factor)
            state = state_p
        # print some log indicates training progress
    print('number of items in q table:', len(model.q_table))
    return model

