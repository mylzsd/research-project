import classifier
import mdp
import reader
import time
import os, sys, random as rd
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from sklearn.neural_network import MLPClassifier


# convert categorical predicted result into binary form
# using one-hot encoding
def class2binary(cluster, num_clf, label_map, data):
    real = data.iloc[:, -1].reset_index(drop = True)
    results = cluster.results(data)
    bi = []
    for i, row in results.iterrows():
        bi_array = [0] * len(label_map) * num_clf
        for j, r in enumerate(row):
            index = j * len(label_map) + label_map[r]
            bi_array[index] = 1
        bi.append(bi_array)
    bi_df = pd.DataFrame(bi)
    return pd.concat([bi_df, real], axis = 1)

def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights[0]), biases[0])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights[1]), biases[1])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights[2]), biases[2])
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights[3]), biases[3])
    layer_4 = tf.nn.sigmoid(layer_4)

    layer_5 = tf.add(tf.matmul(layer_4, weights[4]), biases[4])
    layer_5 = tf.nn.relu(layer_5)

    out_layer = tf.matmul(layer_5, weights[5]) + biases[5]
    return out_layer

def mlp(rdr, dataset, num_clf, hidden_layers, **clf_kwargs):
    # filter out INFO and WARNING logs for tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    print('Dataset', dataset)
    train, train_clf, train_net, test = rdr.read(dataset)
    print(train_clf.shape)
    print(train_net.shape)
    print(test.shape)
    num_feature = train_clf.shape[1] - 1

    print('Training classifiers:', time.asctime(time.localtime(time.time())))
    pure_clf = classifier.Cluster(num_clf, ['rf'], [list(range(num_feature))] * num_clf, **clf_kwargs)
    pure_clf.train(train)
    pure_scores = pure_clf.majorityVote(test)

    features = []
    for i in range(num_clf):
        feature = rd.sample(range(num_feature), int(num_feature / 2))
        features.append(feature)

    cluster = classifier.Cluster(num_clf, ['rf'], features, **clf_kwargs)
    cluster.train(train_clf)
    clf_scores = cluster.majorityVote(test)

    print('Training sci mlp:', time.asctime(time.localtime(time.time())))

    label_map = dict()
    index = 0
    for label in train.iloc[:, -1]:
        if label in label_map: continue
        label_map[label] = index
        index += 1
    for label in test.iloc[:, -1]:
        if label in label_map: continue
        label_map[label] = index
        index += 1
    print('Number of label:', len(label_map))

    train_bi = class2binary(cluster, num_clf, label_map, train_net)
    test_bi = class2binary(cluster, num_clf, label_map, test)
    print(train_bi.shape)
    print(test_bi.shape)

    n_class = len(label_map)
    input_size = n_class * num_clf
    for i, l in enumerate(hidden_layers):
        if l > 1.0:
            hidden_layers[i] = int(l)
        elif l > 0.0:
            hidden_layers[i] = int(l * input_size)
        else:
            hidden_layers[i] = int(-l * n_class)
    hidden_layers = tuple(hidden_layers)
    print(hidden_layers)
    # hidden_layers = (input_size, int(input_size / 2), int(input_size / 2), n_class * 2, n_class)
    mlp = MLPClassifier(hidden_layer_sizes = hidden_layers,
                        alpha = 1e-5, activation = 'relu', solver = 'sgd',
                        random_state = clf_kwargs['random_state'], max_iter = 10000)
    X = train_bi.iloc[:, list(range(n_class * num_clf))]
    y = train_bi.iloc[:, -1]
    mlp.fit(X, y)
    X_test = test_bi.iloc[:, list(range(n_class * num_clf))]
    y_test = test_bi.iloc[:, -1]
    net_score = mlp.score(X_test, y_test)

    # start of tensorflow part

    print('Training tf mlp:', time.asctime(time.localtime(time.time())))

    x = tf.placeholder(tf.float32, shape=(None, input_size))
    y_ = tf.placeholder(tf.float32, shape=(None, n_class))

    x_train_tf = train_bi.iloc[:, list(range(n_class * num_clf))]
    y_train_tf = []
    for i in range(train_bi.shape[0]):
        bi = [0] * n_class
        bi[label_map[train_bi.iloc[i, -1]]] = 1
        y_train_tf.append(bi)
    y_train_tf = pd.DataFrame(y_train_tf)

    x_test_tf = test_bi.iloc[:, list(range(n_class * num_clf))]
    y_test_tf = []
    for i in range(test_bi.shape[0]):
        bi = [0] * n_class
        bi[label_map[test_bi.iloc[i, -1]]] = 1
        y_test_tf.append(bi)
    y_test_tf = pd.DataFrame(y_test_tf)

    hidden_layers = (input_size, int(input_size / 2), int(input_size / 2), n_class * 2, n_class)
    weights = [
                tf.Variable(tf.truncated_normal([input_size, hidden_layers[0]])),
                tf.Variable(tf.truncated_normal([hidden_layers[0], hidden_layers[1]])),
                tf.Variable(tf.truncated_normal([hidden_layers[1], hidden_layers[2]])),
                tf.Variable(tf.truncated_normal([hidden_layers[2], hidden_layers[3]])),
                tf.Variable(tf.truncated_normal([hidden_layers[3], hidden_layers[4]])),
                tf.Variable(tf.truncated_normal([hidden_layers[4], n_class]))
                ]
    biases = [
                tf.Variable(tf.truncated_normal([hidden_layers[0]])),
                tf.Variable(tf.truncated_normal([hidden_layers[1]])),
                tf.Variable(tf.truncated_normal([hidden_layers[2]])),
                tf.Variable(tf.truncated_normal([hidden_layers[3]])),
                tf.Variable(tf.truncated_normal([hidden_layers[4]])),
                tf.Variable(tf.truncated_normal([n_class]))
                ]
    y = multilayer_perceptron(x, weights, biases)

    learning_rate = 0.01
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for e in range(1000):
            sess.run(training_step, feed_dict={x: x_train_tf, y_: y_train_tf})
            cost = sess.run(cost_function, feed_dict={x: x_train_tf, y_: y_train_tf})

            correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            accuracy = sess.run(accuracy, feed_dict={x: x_train_tf, y_: y_train_tf})

            pred_y = sess.run(y, feed_dict={x: x_test_tf})
            mse = tf.reduce_mean(tf.square(pred_y - y_test_tf))
            mse = sess.run(mse)

            if (e + 1) % 100 == 0:
                print('epoch: %d, cost: %f, accuracy: %f, mse: %f' % (e, cost, accuracy, mse))

        correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        test_accu = sess.run(accuracy, feed_dict={x: x_test_tf, y_: y_test_tf})

    # end of tensorflow part
    
    print('full set majority vote: %f' % (pure_scores))
    print('part set majority vote: %f' % (clf_scores))
    print('sci mlp:', net_score)
    print('tf mlp:', test_accu)
    print('Finish time:', time.asctime(time.localtime(time.time())))


def printHelper(results, name, size, verbose):
    results = sorted(results)
    print('%s (%d) (x%d)' % (name, size, len(results)))
    if verbose:
        for res in results:
            print('%s -> %s' % (res[1], res[0]))
    print()


def printResults(pred, real, label_map):
    size = pred.shape[1]
    diff_list = [(list(), 'Single different', size),
                 (list(), 'Multi different', size),
                 (list(), 'Forth quarter', int(size / 4 * 3) + 1),
                 (list(), 'Third quarter', int(size / 2) + 1),
                 (list(), 'Second quarter', int(size / 4) + 1),
                 (list(), 'First quarter', 1),
                 (list(), 'No different', 0)]
    for i in range(pred.shape[0]):
        count = 0
        r = real.iloc[i, real.shape[1] - 1]
        s = set()
        p = list()
        for j in range(pred.shape[1]):
            p.append(label_map[pred.iloc[i, j]])
            s.add(pred.iloc[i, j])
            if pred.iloc[i, j] != r:
                count += 1
        t = (label_map[r], p)
        if count == size:
            if len(s) == 1:
                diff_list[0][0].append(t)
            else:
                diff_list[1][0].append(t)
        elif count >= diff_list[2][2]:
            diff_list[2][0].append(t)
        elif count >= diff_list[3][2]:
            diff_list[3][0].append(t)
        elif count >= diff_list[4][2]:
            diff_list[4][0].append(t)
        elif count > 0:
            diff_list[5][0].append(t)
        else:
            diff_list[6][0].append(t)

    for dl in diff_list:
        printHelper(dl[0], dl[1], dl[2], size <= 20)
    print()


def examine(rdr, dataset, num_clf, reuse, **clf_kwargs):
    print('Dataset', dataset)
    train, train_clf, train_mdp, test = rdr.read(dataset)
    print(train_clf.shape)
    print(train_mdp.shape)
    print(test.shape)

    num_feature = train.shape[1] - 1
    features = [list(range(num_feature))] * num_clf
    cluster = classifier.Cluster(num_clf, ['rf'], features, **clf_kwargs)
    
    label_map = dict()
    index = 65
    for label in train.iloc[:, -1]:
        if label in label_map: continue
        label_map[label] = chr(index)
        index += 1
    for label in test.iloc[:, -1]:
        if label in label_map: continue
        label_map[label] = chr(index)
        index += 1
    print('Number of label:', len(label_map), '\n')

    if reuse:
        cluster.train(train)
        train_res = cluster.results(train)
        printResults(train_res, train, label_map)
        train_scores = cluster.accuracy(train)
        print(train_scores)
        print(np.mean(train_scores))
    else:
        cluster.train(train_clf)
        train_res = cluster.results(train_mdp)
        printResults(train_res, train_mdp, label_map)
        train_scores = cluster.accuracy(train_mdp)
        print(train_scores)
        print(np.mean(train_scores))
    print('\n')

    test_res = cluster.results(test)
    printResults(test_res, test, label_map)

    clf_scores = cluster.accuracy(test)
    print(clf_scores)
    print(np.mean(clf_scores))


def variance(rdr, random_state):
    datasets = ['audiology', 'breast_cancer', 'breast_w', 
                'cmc', 'dematology', 'ecoli', 'glass', 
                'hepatitis', 'human_activity', 'iris', 
                'lymphography']
    n_estimators = [1, 10, 50, 100, 200, 300]
    for d in datasets:
        print('Dataset:', d)
        train, _, _, test = rdr.read(d)
        num_feature = train.shape[1] - 1
        features = [list(range(num_feature))] * 100
        for n in n_estimators:
            kwargs = {'n_estimators': n, 'random_state': random_state}
            cluster = classifier.Cluster(100, ['rf'], features, **kwargs)
            cluster.train(train)
            scores = cluster.accuracy(test)
            print('%03d estimators: mean %f, var %f, max %f, min %f' 
                    % (n, np.mean(scores), np.var(scores), max(scores), min(scores)))
        print('\n')


def majorityVote(rdr, random_state):
    datasets = ['audiology', 'breast_cancer', 'breast_w', 
                'cmc', 'dematology', 'ecoli', 'glass', 
                'hepatitis', 'human_activity', 'iris', 
                'lymphography']
    n_clf = [20, 50, 100, 200, 300]
    for d in datasets:
        print('Dataset:', d)
        train, train_clf, train_mdp, test = rdr.read(d)
        num_feature = train.shape[1] - 1
        for n in n_clf:
            kwargs = {'n_estimators': n, 'random_state': random_state}
            single_rf = classifier.Cluster(1, ['rf'], [list(range(num_feature))], **kwargs)
            single_rf.train(train)
            single_score = single_rf.accuracy(test)[0]

            features = [list(range(num_feature))] * n
            kwargs['n_estimators'] = 1
            cluster = classifier.Cluster(n, ['rf'], features, **kwargs)
            cluster.train(train)
            train2test = cluster.majorityVote(test)
            cluster.train(train_clf)
            half2rest = cluster.majorityVote(train_mdp)
            half2test = cluster.majorityVote(test)
            print('%03d trees\n\tsingle rf: %f\n\ttrain->test: %f\n\tclf->mdp: %f\n\tclf->test: %f'
                    % (n, single_score, train2test, half2rest, half2test))
        print('\n')


def readCommand(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', metavar='MODE', required=True)
    parser.add_argument('-d', '--dataset', metavar='DATASET')
    parser.add_argument('-n', '--num-clf', type=int, default=50)
    parser.add_argument('-u', '--reuse', action='store_true', default=False)
    parser.add_argument('-r', '--random-state', type=int, default=rd.randint(1, 10000))
    parser.add_argument('-s', '--n-estimators', type=int, default=1)
    parser.add_argument('-p', '--portion', type=float, default=0.5)
    parser.add_argument('-l', '--hidden-layers', nargs='+', type=float)
    
    options = parser.parse_args(argv)
    print(options)
    return options


if __name__ == '__main__':
    options = readCommand(sys.argv[1:])
    rs = options.random_state
    portion = options.portion
    rd.seed(rs)
    rdr = reader.Reader(rs, portion)
    clf_kwargs = {'random_state': rs, 'n_estimators': options.n_estimators}
    if options.mode == 'examine':
        examine(rdr, options.dataset, options.num_clf, options.reuse, **clf_kwargs)
    elif options.mode == 'mlp':
        mlp(rdr, options.dataset, options.num_clf, options.hidden_layers, **clf_kwargs)
    elif options.mode == 'variance':
        variance(rdr, rs)
    elif options.mode == 'mjvote':
        majorityVote(rdr, rs)

