from classifier import Cluster
from environment import Environment
import reader
import sys
import random as rd
import argparse
from importlib import import_module


def readCommand(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', metavar='DATASET', required=True)
    parser.add_argument('-a', '--algorithm', metavar='ALGORITHM', required=True)
    parser.add_argument('-n', '--num-clf', type=int, default=50)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001)
    parser.add_argument('-f', '--discount-factor', type=float, default=1.0)
    parser.add_argument('-t', '--num-training', type=int, default=10000)
    parser.add_argument('-e', '--epsilon', type=float, default=0.1)
    parser.add_argument('-r', '--random-state', type=int, default=rd.randint(1, 10000))
    parser.add_argument('-s', '--n-estimators', type=int, default=1)
    parser.add_argument('-p', '--portion', type=float, default=0.5)
    
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
    args['n_estimators'] = options.n_estimators
    args['portion'] = options.portion
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
          n_estimators=1,
          **network_kwargs):
    rd.seed(random_state)

    train, train_clf, train_mdp, test = reader.Reader(random_state, portion).read(dataset)
    print(train_clf.shape)
    print(train_mdp.shape)
    print(test.shape)
    num_feature = train.shape[1] - 1

    label_map = dict()
    for d in [train, test]:
        for l in d.iloc[:, -1]:
            if l not in label_map:
                label_map[l] = len(label_map)
    # single n-tree random forest
    rf = Cluster(1, ['rf'], [list(range(num_feature))], label_map, random_state=random_state, n_estimators=num_clf)
    rf.train(train)
    rf_accu = rf.accuracy(test)[0]
    # multiple single-tree random forests
    features = [list(range(num_feature))] * num_clf
    cluster = Cluster(num_clf, ['rf'], features, label_map, random_state=random_state, n_estimators=n_estimators)
    cluster.train(train_clf)
    mjvote_accu = cluster.majorityVote(test)

    real_set = [train_mdp.iloc[:, -1], test.iloc[:, -1]]
    res_set = [cluster.results(train_mdp), cluster.results(test)]
    prob_set = [cluster.resProb(train_mdp), cluster.resProb(test)]

    sequential = algorithm != 'dqn'
    env = Environment(num_clf, real_set, res_set, prob_set, label_map, sequential=sequential)
    learn = get_learn_function(algorithm)
    model = learn(env, 0, num_training, learning_rate, epsilon, discount_factor, random_state, **network_kwargs)

    # test process, maybe put into env class
    conf_matrix = env.evaluation(model, 1)
    print(conf_matrix)

    print('Single random forest accuracy:', rf_accu)
    print('Majority vote accuracy:', mjvote_accu)
    # print('Reinforcement learning accuracy:', rl_accu)


if __name__ == '__main__':
    options = readCommand(sys.argv[1:])
    train(**options)


