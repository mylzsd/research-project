import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random as rd
from collections import Counter
from environment import Action
import util as U


debug_print = False


class Net(nn.Module):
    def __init__(self, input_size, hiddens, output_size):
        super(Net, self).__init__()
        # using dropout layers
        self.layers = nn.Sequential()
        # self.layers.add_module('in-drop', nn.Dropout(p=0.8))
        for i, hidden in enumerate(hiddens):
            self.layers.add_module('dense' + str(i), nn.Linear(input_size, hidden))
            self.layers.add_module('relu' + str(i), nn.ReLU())
            # self.layers.add_module('lrelu' + str(i), nn.LeakyReLU(negative_slope=0.01))
            self.layers.add_module('dropout' + str(i), nn.Dropout(p=0.5))
            # self.layers.add_module('dropout' + str(i), nn.Dropout(p=0.8))
            input_size = hidden
        self.layers.add_module('out-dense', nn.Linear(input_size, output_size))
        # self.layers = nn.ModuleList()
        # for hidden in hiddens:
        #     self.layers.append(nn.Linear(input_size, hidden))
        #     input_size = hidden
        # self.layers.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        return self.layers(x)
        # for layer in self.layers[:-1]:
        #     x = F.leaky_relu(layer(x), negative_slope=0.1)
        #     # x = F.relu(layer(x))
        # # return F.hardtanh(self.layers[-1](x), min_val=0)
        # return self.layers[-1](x)


class DQN:
    def __init__(self, env, hiddens, learning_rate, discount_factor):
        self.env = env
        self.discount_factor = discount_factor
        input_size = env.num_clf * len(env.label_map)
        self.num_act = env.num_clf + 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(input_size, hiddens, self.num_act)
        # for param in self.net.parameters():
        #     print(param.data)
        self.criterion = nn.MSELoss()
        # self.criterion = nn.MSELoss(reduction='sum')
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)

    def policy(self, state, randomness=0.0):
        self.net.eval()
        actions = self.env.legal_actions(state)
        if len(actions) == 0:
            return None
        # exploration
        if rd.random() < randomness:
            return rd.choice(actions)
        # exploitation
        batch_in = torch.tensor(state.getPred(one_hot=True), dtype=torch.float32)
        q_values = self.net(batch_in).detach().numpy()
        act_val = Counter()
        for a in actions:
            act_val[a] = q_values[a.index]
        return act_val.most_common()[0][0]

    def qValue(self, state):
        self.net.eval()
        actions = self.env.legal_actions(state)
        if len(actions) == 0:
            return 0.0
        batch_in = torch.tensor(state.getPred(one_hot=True), dtype=torch.float32)
        q_values = self.net(batch_in).detach().numpy()
        max_q = -1.0
        for a in actions:
            max_q = max(max_q, q_values[a.index])
        return max_q

    def qValues(self, state):
        self.net.eval()
        batch_in = torch.tensor(state.getPred(one_hot=True), dtype=torch.float32)
        q_values = self.net(batch_in).detach().numpy()
        return q_values

    # def train(self, sample, verbose=False):
    #     self.net.train()
    #     if verbose:
    #         print('\tsample size: %d' % (len(sample)))
    #     # construct training X and y
    #     X = list()
    #     y = list()
    #     for s in sample:
    #         # state: s[0], action: s[1], state_p: s[2], reward: s[3]
    #         if verbose:
    #             print('\n\ts :', str(s[0]), '-> action: ', str(s[1]))
    #             print('\ts\':', str(s[2]), '-> reward: ', s[3])
    #         # get target q value for action
    #         target_q = s[3] + self.discount_factor * self.qValue(s[2])
    #         # get prediction and modify q value for training action
    #         batch_in = torch.tensor(s[0].getPred(one_hot=True), dtype=torch.float32)
    #         q_values = self.net(batch_in).detach().numpy()
    #         target_qs = np.copy(q_values)
    #         target_qs[s[1].index] = target_q
    #         if verbose:
    #             print('\ttarget_q: %.5f index: %d' % (target_q, s[1].index))
    #             print('\tq_values:\n', str(q_values))
    #             print('\ttarget_qs:\n', str(target_qs))
    #         X.append(s[0].getPred(one_hot=True))
    #         y.append(target_qs)
    #     X = torch.tensor(X, dtype=torch.float32)
    #     y = torch.tensor(y, dtype=torch.float32)
    #     # start training
    #     self.optimizer.zero_grad()
    #     y_ = self.net(X)
    #     print(y_.shape)
    #     loss = self.criterion(y_, y)
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss.detach().numpy()

    def train(self, sample, verbose=False):
        self.net.train()
        if verbose:
            print('\tsample size: %d' % (len(sample)))
        # construct training X and y
        X = list()
        y = list()
        oh = list()
        for s in sample:
            # state: s[0], action: s[1], state_p: s[2], reward: s[3]
            if verbose:
                print('\n\ts :', str(s[0]), '-> action: ', str(s[1]))
                print('\ts\':', str(s[2]), '-> reward: ', s[3])
            # get target q value for action
            target_q = s[3] + self.discount_factor * self.qValue(s[2])
            X.append(s[0].getPred(one_hot=True))
            y.append(target_q)
            oh.append(s[1].index)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        # compute output tensor for observed action
        oh = np.eye(self.num_act)[oh]
        oh = torch.tensor(oh, dtype=torch.float32)
        y_ = torch.sum(self.net(X) * oh, dim=1)
        # start training
        self.optimizer.zero_grad()
        loss = self.criterion(y_, y)
        loss.backward()
        self.optimizer.step()
        return loss.detach().numpy()


def learn(env, in_set, num_training, learning_rate, epsilon, discount_factor, random_state, **network_kwargs):
    log_freq = 10
    sample_portion = 0.2
    model = DQN(env, (128, 128, 128), learning_rate, discount_factor)
    num_ins = env.numInstance(in_set)
    sum_loss = 0.0
    # exploration = epsilon
    # decreasing exploration rate
    exploration = 0.6
    fading = 1 / num_training
    for i in range(num_training):
        verbose = False
        sample = list()
        for in_row in range(num_ins):
            state = env.initState()
            history = list()
            while state is not None:
                action = model.policy(state, randomness=exploration)
                state_p, reward = env.step(state, action, in_set, in_row)
                history.append((state, action, state_p, reward))
                state = state_p
            sample.append(history[-1])
            sample.extend(rd.choices(history[:-1], k=int(np.ceil(len(history) * sample_portion))))
        loss = model.train(sample, verbose=(verbose and debug_print))
        sum_loss += loss
        #training log
        if (i + 1) % log_freq == 0:
            print('\nfinished epoch: %d, exploration: %.5f' % (i + 1, exploration))
            print('average loss:', sum_loss / log_freq)
            rl_cmatrix = env.evaluation(model, 1)
            rl_res = U.computeConfMatrix(rl_cmatrix)
            U.outputs(['rl'], [rl_res])
            sum_loss = 0.0
        # decreasing exploration rate
        exploration -= fading
        exploration = max(exploration, 0.1) # keep at least 0.1 exploration rate

    # for i in range(num_training):
    #     in_row = i % num_ins
    #     # verbose = (i + 1) % log_freq == 0
    #     verbose = False
    #     # exploration = np.exp(-i * epsilon)
    #     exploration = epsilon
    #     if verbose:
    #         print('\nepoch: %d row: %d exploration: %.5f' % (i, in_row, exploration))
    #     state = env.initState()
    #     history = list()
    #     if verbose:
    #         actions = list()
    #     while state is not None:
    #         action = model.policy(state, randomness=exploration)
    #         if verbose:
    #             actions.append(str(action))
    #         state_p, reward = env.step(state, action, in_set, in_row)
    #         history.append((state, action, state_p, reward))
    #         state = state_p
    #     if verbose:
    #         print('\t', actions, '\n')

    #     sample = rd.choices(history[:-1], k=int(np.ceil(len(history) * sample_portion)))
    #     sample.append(history[-1])
    #     loss = model.train(sample, verbose=(verbose and debug_print))
    #     sum_loss += loss
    #     if (i + 1) % log_freq == 0:
    #         print('\nfinished epoch', (i + 1))
    #         print('average loss:', sum_loss / log_freq)
    #         rl_cmatrix = env.evaluation(model, 1)
    #         rl_res = U.computeConfMatrix(rl_cmatrix)
    #         U.outputs(['rl'], [rl_res])
    #         sum_loss = 0.0
    return model






