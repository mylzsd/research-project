import classifier
import mdp
import reader
import time
import os, sys, random as rd
import numpy as np
import pandas as pd
import argparse
from sklearn.neural_network import MLPClassifier


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


def mlp(rdr, dataset, num_clf, **clf_kwargs):
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

    print('Training mlp:', time.asctime(time.localtime(time.time())))

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

    input_size = int(len(label_map) * num_clf)
    hidden_layers = (input_size, int(input_size / 2), int(input_size / 2), len(label_map) * 2, len(label_map))
    mlp = MLPClassifier(hidden_layer_sizes = hidden_layers,
                        alpha = 1e-5, activation = 'relu', solver = 'sgd',
                        random_state = clf_kwargs['random_state'], max_iter = 10000)
    X = train_bi.iloc[:, list(range(len(label_map) * num_clf))]
    y = train_bi.iloc[:, -1]
    mlp.fit(X, y)
    X_test = test_bi.iloc[:, list(range(len(label_map) * num_clf))]
    y_test = test_bi.iloc[:, -1]
    net_score = mlp.score(X_test, y_test)
    
    print('full set majority vote: %f' % (pure_scores))
    print('half set majority vote: %f' % (clf_scores))
    print('mlp:', net_score)
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
    parser.add_argument('-s', '--n-estimators', type =int, default=1)
    
    options = parser.parse_args(argv)
    print(options)
    return options


if __name__ == '__main__':
    options = readCommand(sys.argv[1:])
    rs = options.random_state
    rd.seed(rs)
    rdr = reader.Reader(rs)
    clf_kwargs = {'random_state': rs, 'n_estimators': options.n_estimators}
    if options.mode == 'examine':
        examine(rdr, options.dataset, options.num_clf, options.reuse, **clf_kwargs)
    elif options.mode == 'mlp':
        mlp(rdr, options.dataset, options.num_clf, **clf_kwargs)
    elif options.mode == 'variance':
        variance(rdr, rs)
    elif options.mode == 'mjvote':
        majorityVote(rdr, rs)

