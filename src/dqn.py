import numpy as np
import random as rd
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import Counter
from environment import Action


debug_print = False


def ann(input_, hiddens, num_act):
    out = input_
    for hidden in hiddens:
        bias = tf.constant(1, dtype=np.float64, shape=(1, 1))
        out = tf.concat([out, bias], 1)
        out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
        # out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
        # out = tf.nn.leaky_relu(out, 0.8)
    out = layers.fully_connected(out, num_outputs=num_act, activation_fn=None)
    return out


class DQN:
    def __init__(self, env, hiddens, learning_rate, discount_factor):
        self.env = env
        self.discount_factor = discount_factor
        input_size = env.num_clf * len(env.label_map)
        self.num_act = env.num_clf + 1
        self.input_state = tf.placeholder(np.float64, shape=(None, input_size))
        # compose deep Q network
        self.network = ann(self.input_state, hiddens, self.num_act)
        self.target = tf.placeholder(np.float64, None)
        self.loss = tf.losses.mean_squared_error(labels=self.target, predictions=self.network)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training = self.optimizer.minimize(self.loss)
        # start a tensorflow session and initialize variables
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def policy(self, state, randomness=0.0):
        actions = self.env.legal_actions(state)
        if len(actions) == 0:
            return None
        # exploration
        if rd.random() < randomness:
            return rd.choice(actions)
        # exploitation
        q_values = self.sess.run(self.network, feed_dict={self.input_state: state.getPred(one_hot=True)})
        act_val = Counter()
        for a in actions:
            act_val[a] = q_values[0, a.index]
        return act_val.most_common()[0][0]

    def qValue(self, state):
        actions = self.env.legal_actions(state)
        if len(actions) == 0:
            return 0.0
        q_values = self.sess.run(self.network, feed_dict={self.input_state: state.getPred(one_hot=True)})
        max_q = -1.0
        for a in actions:
            max_q = max(max_q, q_values[0, a.index])
        return max_q

    def train(self, state, action, state_p, reward, in_set, in_row):
        if debug_print:
            print('\n\tstate:', str(state), '-> action: ', str(action))
            print('\tstate_p:', str(state_p), '-> reward: ', reward)
        # get target q value for action
        target_q = reward + self.discount_factor * self.qValue(state_p)
        # get prediction and modify q value for training action
        feed_dict = {self.input_state: state.getPred(one_hot=True)}
        q_values = self.sess.run(self.network, feed_dict=feed_dict)
        target_qs = np.array(q_values[0])
        target_qs[action.index] = target_q
        if debug_print:
            print('\ttarget_q:', target_q, 'index:', action.index)
            print('\tq_values:', str(q_values[0]), q_values[0].shape)
            print('\ttarget_qs:', str(target_qs), target_qs.shape)
        # run training step
        feed_dict[self.target] = target_qs
        _, loss = self.sess.run([self.training, self.loss], feed_dict=feed_dict)
        return loss

    def close(self):
        self.sess.close()


def learn(env, in_set, num_training, learning_rate, epsilon, discount_factor, random_state, **network_kwargs):
    log_freq = 100
    batch_size = 10
    model = DQN(env, (128, 128, 64, 64), learning_rate, discount_factor)
    num_ins = env.numInstance(in_set)
    total_loss = 0.0
    total_sample_size = 0
    for i in range(num_training):
        in_row = i % num_ins
        if debug_print:
            print('epoch:', i, 'row:', in_row)
        history = []
        state = env.initState()
        if debug_print:
            actions = []
        while state is not None:
            action = model.policy(state, randomness=epsilon)
            if debug_print:
                actions.append(str(action))
            state_p, reward = env.step(state, action, in_set, in_row)
            history.append((state, action, state_p, reward))
            state = state_p
        if debug_print:
            print('\t', actions)

        # sample = rd.choices(history, k=batch_size)
        sample = history
        total_sample_size += len(sample)
        for s in sample:
            l = model.train(s[0], s[1], s[2], s[3], in_set, in_row)
            total_loss += l

        if (i + 1) % log_freq == 0:
            print('finished epoch', (i + 1))
            print('average loss:', total_loss / total_sample_size)
            # print(env.evaluation(model, 1))
            total_loss = 0.0
            total_sample_size = 0
    return model
    


