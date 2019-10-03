from classifier import Cluster
from environment import Environment
import reader as rdr
import util as U
import sys
import random as rd
import numpy as np
import argparse
from importlib import import_module
from sklearn.model_selection import KFold


def readCommand(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', metavar='DATASET', required=True)
    parser.add_argument('-a', '--algorithm', metavar='ALGORITHM', required=True)
    parser.add_argument('-n', '--num-clf', type=int, default=50)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-f', '--discount-factor', type=float, default=1)
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

    data = rdr.read(dataset)
    print(data.shape)
    # shuffle dataset
    data = data.sample(frac=1).reset_index(drop=True)

    kf = KFold(n_splits=10)
    for i, (train_idx, test_idx) in enumerate(kf.split(data)):
        print('Running iteration %d of 10 fold...' % (i + 1))
        train = data.iloc[train_idx, :]
        test = data.iloc[test_idx, :]
        train_clf, train_mdp = rdr.splitByPortion(train, portion, random_state)
        # print(train_clf.shape)
        # print(train_mdp.shape)
        # print(test.shape)
        num_feature = train.shape[1] - 1

        label_map = dict()
        # label_map[None] = 'N'
        for d in [train, test]:
            for l in d.iloc[:, -1]:
                if l not in label_map:
                    label_map[l] = len(label_map)
        
        features = list()
        for _ in range(num_clf):
            features.append(list(range(num_feature)))
            # features.append(rd.choices(list(range(num_feature)), k=int(np.ceil(num_feature * 0.5))))
        # features = [list(range(num_feature))] * num_clf
        bm_cluster = Cluster(num_clf, ['dt'], features, label_map, random_state=random_state)
        bm_cluster.train(train)
        mv_cmatrix = bm_cluster.majorityVote(test)
        wv_cmatrix = bm_cluster.weightedVote(test)
        rl_cluster = Cluster(num_clf, ['dt'], features, label_map, random_state=random_state)
        rl_cluster.train(train_clf)
        train_accu = rl_cluster.accuracy(train_mdp)
        test_accu = rl_cluster.accuracy(test)

        real_set = [train_mdp.iloc[:, -1], test.iloc[:, -1]]
        res_set = [rl_cluster.results(train_mdp), rl_cluster.results(test)]
        prob_set = [rl_cluster.resProb(train_mdp), rl_cluster.resProb(test)]

        env = Environment(num_clf, real_set, res_set, prob_set, label_map, sequential=sequential)
        learn = get_learn_function(algorithm)
        model = learn(env, 0, num_training, learning_rate, epsilon, discount_factor, random_state, **network_kwargs)
        rl_cmatrix = env.evaluation(model, 1, verbose=True)

        # print(mv_cmatrix)
        # print(wv_cmatrix)
        # print(rl_cmatrix)
        mv_res = U.computeConfMatrix(mv_cmatrix)
        wv_res = U.computeConfMatrix(wv_cmatrix)
        rl_res = U.computeConfMatrix(rl_cmatrix)
        U.outputs(['mv', 'wv', 'rl'], [mv_res, wv_res, rl_res])
        print(U.formatFloats(train_accu, 2))
        print(U.formatFloats(test_accu, 2))
        break


if __name__ == '__main__':
    options = readCommand(sys.argv[1:])
    train(**options)


