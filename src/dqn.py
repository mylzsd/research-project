import numpy as np
import random as rd
import tensorflow as tf
import tensorflow.contrib.layers as layers
from environment import Action


def policy_net(input_, hiddens, num_act, activation):
    out = input_
    for hidden in hiddens:
        if activation == 'logistic':
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.sigmoid)
        elif activation == 'tanh':
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.tanh)
        elif activation == 'relu':
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, num_outputs=num_act, activation_fn=tf.nn.softmax)
    return out


def value_net(input_, hiddens, activation):
    out = input_
    for hidden in hiddens:
        if activation == 'logistic':
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.sigmoid)
        elif activation == 'tanh':
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.tanh)
        elif activation == 'relu':
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, num_outputs=1, activation_fn=None)
    return out


class DQN:
    def __init__(self, env, policy_hiddens, value_hiddens, activation):
        self.env = env
        input_size = env.num_clf * len(env.label_map)
        self.num_act = env.num_clf + 1
        self.input_ = tf.placeholder(tf.float32, shape=(None, input_size))
        self.p_net = policy_net(self.input_, policy_hiddens, self.num_act, activation)
        self.v_net = value_net(self.input_, value_hiddens, activation)
        print(self.p_net.shape)
        print(self.v_net.shape)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def a2v(self, action):
        v = np.zeros(self.num_act, dtype=float)
        v[action.index] = 1
        return np.reshape(v, (1, -1))

    def policy(self, state, randomness=0.0):
        actions = self.env.legal_actions(state)
        if len(actions) == 0:
            return None
        # exploration or exploitation
        if rd.random() < randomness:
            return rd.choice(actions)
        act_prob = self.sess.run(self.p_net, feed_dict={self.input_: state.getPred(one_hot=True)})
        max_prob = 0.0
        candidates = []
        for a in actions:
            prob = act_prob[0, a.index]
            if prob >= max_prob:
                if prob > max_prob:
                    max_prob = prob
                    candidates.clear()
                candidates.append((a, prob))
        return rd.choice(candidates)[0]

    def qValue(self, state):
        if state is None:
            return np.zeros((1, 1), dtype=float)
        value = self.sess.run(self.v_net, feed_dict={self.input_: state.getPred(one_hot=True)})
        return value

    def train(self, state, action, state_p, reward, learning_rate, discount_factor, in_set, in_row):
        # first update policy network
        actions = self.env.legal_actions(state)
        max_q = float('-inf')
        candidates = []
        for a in actions:
            n_state, reward = self.env.step(state, a, in_set, in_row)
            q = reward + discount_factor * self.qValue(n_state)[0, 0]
            if q >= max_q:
                if q > max_q:
                    max_q = q
                    candidates.clear()
                candidates.append(a)
        target_act = rd.choice(candidates)
        target_act = self.a2v(target_act)
        cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_act, logits=self.p_net))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_step = optimizer.minimize(cost_function)
        cost, _ = self.sess.run([cost_function, training_step], feed_dict={self.input_: state.getPred(one_hot=True)})

        # then update value network
        target_v = tf.constant(reward + discount_factor * self.qValue(state_p), dtype=tf.float32)
        loss = tf.losses.absolute_difference(labels=target_v, predictions=self.v_net)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_step = optimizer.minimize(loss)
        loss, _ = self.sess.run([loss, training_step], feed_dict={self.input_: state.getPred(one_hot=True)})

        return (cost, loss)

    def close(self):
        self.sess.close()


def learn(env, in_set, num_training, learning_rate, epsilon, discount_factor, random_state, **network_kwargs):
    log_freq = 10
    batch_size = 0.1
    model = DQN(env, (64, 64, 32), (64, 32), 'relu')
    num_ins = env.numInstance(in_set)
    for i in range(num_training):
        in_row = i % num_ins
        history = []
        state = env.initState()
        while state is not None:
            action = model.policy(state)
            state_p, reward = env.step(state, action, in_set, in_row)
            history.append((state, action, state_p, reward))
            state = state_p

        total_cost = 0.0
        total_loss = 0.0
        sample = [history[-1]]
        if len(history) > 1:
            sample += rd.choices(history[:-1], k=int(batch_size * (env.num_clf - 1)))
        for s in sample:
            c, l = model.train(s[0], s[1], s[2], s[3], learning_rate, discount_factor, in_set, in_row)
            total_cost += c
            total_loss += l

        if (i + 1) % log_freq == 0:
            print('finished epoch', (i + 1))
            print('average policy cost:', total_cost / log_freq)
            print('average value loss:', total_loss / log_freq)
            print(env.evaluation(model, 1))
    return model





