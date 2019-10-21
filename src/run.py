from classifier import Cluster
from environment import Environment
import fs
import reader as rdr
import util as U
import sys
import random as rd
import numpy as np
import argparse
import time
from collections import Counter
from importlib import import_module
from sklearn.model_selection import KFold


def readCommand(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', metavar='DATASET', required=True)
    parser.add_argument('-a', '--algorithm', metavar='ALGORITHM', required=True)
    parser.add_argument('-n', '--num-clf', type=int, default=50)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.1)
    parser.add_argument('-f', '--discount-factor', type=float, default=1.0)
    parser.add_argument('-t', '--num-training', type=int, default=10000)
    parser.add_argument('-e', '--epsilon', type=float, default=0.1)
    parser.add_argument('-r', '--random-state', type=int, default=rd.randint(1, 10000))
    parser.add_argument('-p', '--portion', type=float, default=0.5)
    parser.add_argument('-s', '--sequential', type=bool, default=True)
    
    options = parser.parse_args(argv)
    args = dict()
    args['dataset'] = options.dataset
    args['algorithm'] = options.algorithm
    args['num_clf'] = options.num_clf
    args['num_training'] = options.num_training
    args['learning_rate'] = options.learning_rate
    args['discount_factor'] = options.discount_factor
    args['epsilon'] = options.epsilon
    args['random_state'] = options.random_state
    args['portion'] = options.portion
    if options.algorithm in ['alphanet', 'tfdqn', 'ptdqn']:
        args['sequential'] = False
    else:
        args['sequential'] = options.sequential
    print(args)
    return args


def get_learn_function(alg):
    # alg_module = import_module('.'.join(['src', alg]))
    alg_module = import_module(alg)
    return alg_module.learn


def train(dataset,
          algorithm,
          random_state,
          num_clf=50,
          num_training=10000,
          learning_rate=0.1,
          discount_factor=1.0,
          epsilon=1.0,
          portion=0.5,
          sequential=True,
          **network_kwargs):
    rd.seed(random_state)
    np.random.seed(random_state)
    start_time = time.time()

    data = rdr.read(dataset)
    print(data.shape)
    # shuffle dataset
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    data_reading_time = time.time() - start_time
    print('reading data takes %.3f sec' % (data_reading_time))

    num_feature = data.shape[1] - 1
    label_map = dict()
    # label_map[None] = 'N'
    for l in data.iloc[:, -1]:
        if l not in label_map:
            label_map[l] = len(label_map)
    print('number of labels: %d' % (len(label_map)))

    feature_type = 3
    features = list()
    for i in range(num_clf):
        if feature_type == 1:
            features.append(list(range(num_feature)))
        elif feature_type == 2:
            features.append(rd.choices(list(range(num_feature)), k=int(np.ceil(num_feature * 0.5))))
        elif feature_type == 3:
            # first 1/3 features for first 1/3 clf, second for second, and third for third
            size = int((num_feature - 1) / 3) + 1
            index = int((num_clf - 1) / 3) + 1
            if i < index:
                features.append(list(range(size)))
            elif i < 2 * index:
                features.append(list(range(size, 2 * size)))
            else:
                features.append(list(range(2 * size, num_feature)))
    # features = [list(range(num_feature))] * num_clf
    # print(features)

    clf_type = 1
    if clf_type == 1:
        clf_types = ['dt']
    elif clf_type == 2:
        clf_types = ['mlp']
    elif clf_type == 3:
        clf_types = ['knn']
    elif clf_type == 4:
        clf_types = ['nb']
    elif clf_type == 5:
        clf_types = ['dt', 'mlp', 'knn', 'nb']
    # print(clf_types)

    mv_stat = [0.0] * 4
    wv_stat = [0.0] * 4
    fs_stat = [0.0] * 4
    rl_stat = [0.0] * 4
    fs_size = 0.0
    rl_size = 0.0
    avg_full_test_accu = 0.0
    avg_test_accu = 0.0

    term = 3
    kf = KFold(n_splits=10, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(kf.split(data)):
        print('\nRunning iteration %d of 10 fold...' % (i + 1))
        # print(test_idx)
        out_model = []
        out_res = []
        train = data.iloc[train_idx, :]
        test = data.iloc[test_idx, :]
        train_clf, train_ens = rdr.splitByPortion(train, portion, random_state)
        train_dqn, valid_dqn = rdr.splitByPortion(train_ens, 0.5, random_state)
        # print(train_clf.shape)
        # print(train_ens.shape)
        # print(test.shape)
        mdp_label_count = Counter()
        for l in train_ens.iloc[:, -1]:
            mdp_label_count[l] += 1
        test_label_count = Counter()
        for l in test.iloc[:, -1]:
            test_label_count[l] += 1
        
        # results for majority vote and weighted vote
        bm_cluster = Cluster(num_clf, clf_types, features, label_map, random_state=random_state)
        bm_cluster.train(train)
        # bm_cluster.train(train_clf)
        full_test_accu = bm_cluster.accuracy(test)
        avg_full_test_accu += np.mean(full_test_accu)

        mv_cmatrix = bm_cluster.majorityVote(test)
        # print(mv_cmatrix)
        mv_res = U.computeConfMatrix(mv_cmatrix)
        for s in range(4):
            mv_stat[s] += mv_res[s]
        out_model.append('mv')
        out_res.append(mv_res)

        wv_cmatrix = bm_cluster.weightedVote(test)
        # print(wv_cmatrix)
        wv_res = U.computeConfMatrix(wv_cmatrix)
        for s in range(4):
            wv_stat[s] += wv_res[s]
        out_model.append('wv')
        out_res.append(wv_res)

        # classifiers trained by half data
        rl_cluster = Cluster(num_clf, clf_types, features, label_map, random_state=random_state)
        rl_cluster.train(train_clf)
        train_accu = rl_cluster.accuracy(train_ens)
        valid_accu = rl_cluster.accuracy(valid_dqn)
        test_accu = rl_cluster.accuracy(test)
        avg_test_accu += np.mean(test_accu)

        real_set = [train_ens.iloc[:, -1], test.iloc[:, -1], train_dqn.iloc[:, -1], valid_dqn.iloc[:, -1]]
        res_set = [rl_cluster.results(train_ens), rl_cluster.results(test), rl_cluster.results(train_dqn), rl_cluster.results(valid_dqn)]
        prob_set = [rl_cluster.resProb(train_ens), rl_cluster.resProb(test), rl_cluster.resProb(train_dqn), rl_cluster.resProb(valid_dqn)]
        label_count_set = []

        # FS algorithm
        fs_model = fs.train(num_clf, real_set[0], res_set[0])
        print(fs_model)
        fs_size += len(fs_model)
        fs_cmatrix = fs.evaluation(fs_model, real_set[1], res_set[1], label_map)
        # print(fs_cmatrix)
        fs_res = U.computeConfMatrix(fs_cmatrix)
        for s in range(4):
            fs_stat[s] += fs_res[s]
        out_model.append('fs')
        out_res.append(fs_res)

        # CBRL model
        model_path = 'models/d{}n{:d}c{:d}f{:d}r{:d}i{:d}.ris'.format(dataset, num_clf, clf_type, feature_type, random_state, i)
        # print(model_path)
        env = Environment(num_clf, real_set, res_set, prob_set, label_map, label_count_set, sequential=sequential)
        learn = get_learn_function(algorithm)
        model = learn(env, 0, num_training, learning_rate, epsilon, discount_factor, random_state, **network_kwargs)
        # model = learn(env, 2, num_training, learning_rate, epsilon, discount_factor, random_state, **network_kwargs)
        model.save(model_path)
        # model.load(model_path)
        rl_cmatrix, avg_clf = env.evaluation(model, 1, verbose=False)
        rl_size += avg_clf
        print('%.2f classifiers used' % (avg_clf))
        # print(rl_cmatrix)
        rl_res = U.computeConfMatrix(rl_cmatrix)
        for s in range(4):
            rl_stat[s] += rl_res[s]
        out_model.append('rl')
        out_res.append(rl_res)

        U.outputs(out_model, out_res)
        print(np.mean(train_accu))
        print(U.formatFloats(train_accu, 2) + '\n')
        print(np.mean(valid_accu))
        print(U.formatFloats(valid_accu, 2) + '\n')
        print(np.mean(test_accu))
        print(U.formatFloats(test_accu, 2) + '\n')
        print(np.mean(full_test_accu))
        print(U.formatFloats(full_test_accu, 2) + '\n')
        if i >= term - 1:
            break

    mv_stat = [n / term for n in mv_stat]
    wv_stat = [n / term for n in wv_stat]
    fs_stat = [n / term for n in fs_stat]
    rl_stat = [n / term for n in rl_stat]
    fs_size /= term
    rl_size /= term
    avg_full_test_accu /= term
    avg_test_accu /= term
    U.outputs(['mv', 'wv', 'fs', 'rl'], [mv_stat, wv_stat, fs_stat, rl_stat])
    print('fs avg size: %.5f, rl avg size: %.5f' % (fs_size, rl_size))
    print('full test avg accu: %.5f, test avg accu: %.5f' % (avg_full_test_accu, avg_test_accu))
    training_time = time.time() - start_time - data_reading_time
    print('\ntraining takes %.3f sec' % (training_time))


if __name__ == '__main__':
    options = readCommand(sys.argv[1:])
    train(**options)


