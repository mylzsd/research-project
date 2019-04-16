import reader
import random as rd
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.layers as layers
from sklearn.neural_network import MLPClassifier


def mlp(input_, hiddens, num_labels, activation):
    out = input_
    for hidden in hiddens:
        if activation == 'logistic':
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.sigmoid)
        elif activation == 'tanh':
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.tanh)
        elif activation == 'relu':
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, num_outputs=num_labels, activation_fn=tf.nn.sigmoid)
    return out


def train_fn(sess, train_op, feed_dict):
    sess.run(train_op, feed_dict=feed_dict)


def main():
    r_state = rd.randint(1, 1000)
    # r_state = 480
    rd.seed(r_state)
    dataset = 'breast_cancer'
    # activation = 'logistic'
    # activation = 'tanh'
    activation = 'relu'
    solver = 'adam'
    # solver = 'sgd'

    train, _, _, test = reader.Reader(r_state).read(dataset)

    label_map = dict()
    for l in train.iloc[:, -1]:
        if l not in label_map:
            label_map[l] = len(label_map)
    for l in test.iloc[:, -1]:
        if l not in label_map:
            label_map[l] = len(label_map)
    num_labels = len(label_map)

    x_train = train.iloc[:, :-1]
    y_train = []
    for l in train.iloc[:, -1]:
        y_temp = [0.0] * num_labels
        y_temp[label_map[l]] = 1.0
        y_train.append(y_temp)
    y_train = pd.DataFrame(y_train)

    x_test = test.iloc[:, :-1]
    y_test = []
    for l in test.iloc[:, -1]:
        y_temp = [0.0] * num_labels
        y_temp[label_map[l]] = 1.0
        y_test.append(y_temp)
    y_test = pd.DataFrame(y_test)

    hiddens = (32, 16)

    x = tf.placeholder(tf.float32, shape=(None, train.shape[1] - 1))
    y_ = tf.placeholder(tf.float32, shape=(None, num_labels))
    y = mlp(x, hiddens, num_labels, activation)
    # cost_function = tf.reduce_mean(tf.square(y_ - y))
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

    if solver == 'adam':
        optimizer = tf.train.AdamOptimizer()
    elif solver == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(0.001)
    training_step = optimizer.minimize(cost_function)

    # init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        x_batch = x_train
        y_batch = y_train
        train_fn(sess, training_step, {x: x_batch, y_: y_batch})
        # sess.run(training_step, feed_dict={x: x_batch, y_: y_batch})
        if (i + 1) % 1000 == 0:
            # cost = sess.run(cost_function, feed_dict={x: x_batch, y_: y_batch})
            correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            cost, training_accu = sess.run([cost_function, accuracy], feed_dict={x: x_batch, y_: y_batch})
            testing_accu = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
            print('epoch: %d, cost: %f, training_accu: %f, testing_accu: %f' % (i, cost, training_accu, testing_accu))
    tf_score = testing_accu
    sess.close()

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     for i in range(10000):
    #         x_batch = x_train
    #         y_batch = y_train
    #         # cand = rd.choices(list(range(train.shape[0])), k=50)
    #         # x_batch = x_train.iloc[cand]
    #         # y_batch = y_train.iloc[cand]
    #         sess.run(training_step, feed_dict={x: x_batch, y_: y_batch})
    #         if (i + 1) % 1000 == 0:
    #             cost = sess.run(cost_function, feed_dict={x: x_batch, y_: y_batch})

    #             correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #             accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    #             training_accu = sess.run(accuracy, feed_dict={x: x_batch, y_: y_batch})
    #             testing_accu = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})

    #             # pred_y = sess.run(y, feed_dict={x: x_test})
    #             # mse = tf.reduce_mean(tf.square(pred_y - y_test))
    #             # mse = sess.run(mse)
    #             # print('epoch: %d, cost: %f, accuracy: %f, mse: %f' % (i, cost, accuracy, mse))
    #             print('epoch: %d, cost: %f, training_accu: %f, testing_accu: %f' % (i, cost, training_accu, testing_accu))

    #     # correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #     # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    #     # tf_score = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
    #     tf_score = testing_accu

    sl_nn = MLPClassifier(hidden_layer_sizes=hiddens, activation=activation, solver=solver, max_iter=10000, random_state=r_state)
    sl_nn.fit(x_train, y_train)
    sl_score = sl_nn.score(x_test, y_test)

    print('dataset: %s\ntraining: %s\ntesting: %s\n# labels: %d\nrandom state: %d\nactivation: %s\nsolver: %s'
            % (dataset, str(train.shape), str(test.shape), num_labels, r_state, activation, solver))
    print('tf: %f\nsl: %f' % (tf_score, sl_score))



if __name__ == '__main__':
    main()





