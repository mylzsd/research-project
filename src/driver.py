import classifier
import mdp
import reader
import sys
import numpy as np
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
    parser.add_argument('-s', '--n-estimators', type=int, default=100)
    
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
    print(args)
    return args


def run(dataset,
        model,
        num_clf=50,
        num_training=10000,
        learning_rate=0.1,
        discount_factor=1.0,
        epsilon=1.0,
        **kwargs):
    train, train_clf, train_mdp, test = reader.Reader(kwargs['random_state']).read(dataset)
    print(train_clf.shape)
    print(train_mdp.shape)
    print(test.shape)
    num_feature = train_clf.shape[1] - 1

    rf_kwargs = {'random_state': kwargs['random_state'], 'n_estimators': 300}
    rf = classifier.Cluster(1, ['rf'], [list(range(num_feature))], **rf_kwargs)
    rf.train(train)
    rf_accu = rf.accuracy(test)[0]

    features = [list(range(num_feature))] * num_clf
    cluster = classifier.Cluster(num_clf, ['rf'], features, **kwargs)
    cluster.train(train_clf)
    mjvote_accu = cluster.majorityVote(test)

    rl = mdp.MDP(cluster, model, learning_rate, discount_factor, epsilon, kwargs['random_state'])
    rl.train(train_mdp, num_training, test)
    mdp_accu = rl.accuracy(test)

    print('Single random forest accuracy:', rf_accu)
    print('Majority vote accuracy:', mjvote_accu)
    print('Reinforcement learning accuracy:', mdp_accu)


if __name__ == '__main__':
    options = readCommand(sys.argv[1:])
    run(**options)

