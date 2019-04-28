import numpy as np
import random as rd
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import Counter
from environment import Action


def policy_net(input_, hiddens, num_act):
    out = input_
    for hidden in hiddens:
        out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, num_outputs=num_act, activation_fn=None)
    return out


def value_net(input_, hiddens):
    out = input_
    for hidden in hiddens:
        out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, num_outputs=1, activation_fn=None)
    return out


class DQN:
    def __init__(self, env, value_hiddens, policy_hiddens, learning_rate, discount_factor):
        self.env = env
        self.discount_factor = discount_factor
        input_size = env.num_clf * len(env.label_map)
        self.num_act = env.num_clf + 1
        self.input_state = tf.placeholder(np.float64, shape=(None, input_size))
        # components of value net
        self.v_net = value_net(self.input_state, value_hiddens)
        self.v_target = tf.placeholder(np.float64, None)
        self.v_loss = tf.losses.absolute_difference(labels=self.v_target, predictions=self.v_net)
        self.v_optimizer = tf.train.AdamOptimizer(learning_rate)
        self.v_training = self.v_optimizer.minimize(self.v_loss)
        # components of policy net
        self.p_net = policy_net(self.input_state, policy_hiddens, self.num_act)
        self.p_target = tf.placeholder(np.float64, shape=(None, self.num_act))
        self.p_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.p_target, logits=self.p_net))
        self.p_optimizer = tf.train.AdamOptimizer(learning_rate)
        self.p_training = self.p_optimizer.minimize(self.p_loss)
        # start a tensorflow session and initialize variables
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def a2v(self, action):
        v = np.zeros(self.num_act, dtype=np.float64)
        v[action.index] = 1.0
        return np.reshape(v, (1, -1))

    def policy(self, state, in_set, in_row, randomness=0.0):
        actions = self.env.legal_actions(state)
        if len(actions) == 0:
            return None
        # exploration
        if rd.random() < randomness:
            return rd.choice(actions)
        # exploitation
        act = self.sess.run(self.p_net, feed_dict={self.input_state: state.getPred(one_hot=True)})
        act_prob = Counter()
        for a in actions:
            act_prob[a] = act[0, a.index]
        return act_prob.most_common()[0][0]

    def qValue(self, state):
        if state is None:
            return 0.0
        value = self.sess.run(self.v_net, feed_dict={self.input_state: state.getPred(one_hot=True)})[0][0]
        return value

    def train(self, state, action, state_p, reward, in_set, in_row):
        # update policy net
        actions = self.env.legal_actions(state)
        act_value = Counter()
        for a in actions:
            state_p, reward = self.env.step(state, a, in_set, in_row)
            q = reward + self.discount_factor * self.qValue(state_p)
            act_value[a] = q
        target_action = self.a2v(act_value.most_common()[0][0])
        feed_dict = {
            self.input_state: state.getPred(one_hot=True),
            self.p_target: target_action
        }
        p_net_loss, _ = self.sess.run([self.p_loss, self.p_training], feed_dict=feed_dict)
        # update value net
        real_q = reward + self.discount_factor * self.qValue(state_p)
        feed_dict = {
            self.input_state: state.getPred(one_hot=True),
            self.v_target: real_q
        }
        v_net_loss, _ = self.sess.run([self.v_loss, self.v_training], feed_dict=feed_dict)
        return (p_net_loss, v_net_loss)

    def close(self):
        self.sess.close()


def learn(env, in_set, num_training, learning_rate, epsilon, discount_factor, random_state, **network_kwargs):
    log_freq = 1000
    batch_size = 20
    model = DQN(env, (128, 64, 32), (128, 64, 32), learning_rate, discount_factor)
    num_ins = env.numInstance(in_set)
    total_p_loss = 0.0
    total_v_loss = 0.0
    for i in range(num_training):
        in_row = i % num_ins
        history = []
        state = env.initState()
        while state is not None:
            action = model.policy(state, in_set, in_row, randomness=epsilon)
            state_p, reward = env.step(state, action, in_set, in_row)
            history.append((state, action, state_p, reward))
            state = state_p

        sample = rd.choices(history, k=batch_size)
        for s in sample:
            p, v = model.train(s[0], s[1], s[2], s[3], in_set, in_row)
            total_p_loss += p
            total_v_loss += v

        if (i + 1) % log_freq == 0:
            print('finished epoch', (i + 1))
            print('average policy loss:', total_p_loss / (log_freq * batch_size))
            print('average value loss:', total_v_loss / (log_freq * batch_size))
            # print(env.evaluation(model, 1))
            total_p_loss = 0.0
            total_v_loss = 0.0
    return model





