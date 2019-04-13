from classifier import Cluster
from environment import Environment
import mdp
import reader
import sys
import random as rd
import argparse


def readCommand(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', metavar='DATASET', required=True)
    parser.add_argument('-m', '--model', metavar='MODEL', required=True)
    parser.add_argument('-n', '--num-clf', type=int, default=50)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.1)
    parser.add_argument('-f', '--discount-factor', type=float, default=1.0)
    parser.add_argument('-t', '--num-training', type=int, default=10000)
    parser.add_argument('-e', '--epsilon', type=float, default=0.1)
    parser.add_argument('-r', '--random-state', type=int, default=rd.randint(1, 10000))
    parser.add_argument('-s', '--n-estimators', type=int, default=1)
    parser.add_argument('-p', '--portion', type=float, default=0.5)
    
    options = parser.parse_args(argv)
    args = dict()
    args['dataset'] = options.dataset
    args['model'] = options.model
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


def train(dataset,
          model,
          num_clf=50,
          num_training=10000,
          learning_rate=0.1,
          discount_factor=1.0,
          epsilon=1.0,
          portion=0.5,
          **kwargs):
    rd.seed(kwargs['random_state'])

    train, train_clf, train_mdp, test = reader.Reader(kwargs['random_state'], portion).read(dataset)
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
    rf_kwargs = {'random_state': kwargs['random_state'], 'n_estimators': num_clf}
    rf = Cluster(1, ['rf'], [list(range(num_feature))], label_map, **rf_kwargs)
    rf.train(train)
    rf_accu = rf.accuracy(test)[0]
    # multiple single-tree random forests
    features = [list(range(num_feature))] * num_clf
    cluster = Cluster(num_clf, ['rf'], features, label_map, **kwargs)
    cluster.train(train_clf)
    mjvote_accu = cluster.majorityVote(test)

    real_set = [train_mdp.iloc[:, -1], test.iloc[:, -1]]
    res_set = [cluster.results(train_mdp), cluster.results(test)]
    prob_set = [cluster.resProb(train_mdp), cluster.resProb(test)]

    env = Environment(num_clf, real_set, res_set, prob_set, label_map)
    # q_func = TODO: some load module

    for i in range(num_training):
        for r in rd.sample(list(range(train_mdp.shape[0])), 100):
            state = env.initState()

    '''
    TODO: training process
    for n epoch
        select batch-size samples
        for each sample
            get initial state
            while not final state
                get action from q function
                apply action 
                update q function using new state and reward
    '''
    '''
    TODO: test process
    for each instance in test set
        get initial state
        while not final state
            get action from q function and apply
        get predicted class and real class
    compute accuracy and other measurement based on prediction matrix
    '''

    # rl = mdp.MDP(cluster, model, learning_rate, discount_factor, epsilon, kwargs['random_state'])
    # rl.train(train_mdp, num_training, test)
    # mdp_accu = rl.accuracy(test)

    # print('Single random forest accuracy:', rf_accu)
    # print('Majority vote accuracy:', mjvote_accu)
    # print('Reinforcement learning accuracy:', mdp_accu)


if __name__ == '__main__':
    options = readCommand(sys.argv[1:])
    train(**options)


