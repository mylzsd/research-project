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
    parser.add_argument('-l', '--learning-rate', type=float, default=0.01)
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
    if options.algorithm in ['alphanet', 'dqn']:
        args['sequential'] = False
    else:
        args['sequential'] = options.sequential
    print(args)
    return args


def get_learn_function(alg):
    # alg_module = import_module('.'.join(['src', alg]))
    alg_module = import_module(alg)
    return alg_module.learn


def computeConfMatrix(conf_matrix):
    total_count = conf_matrix.sum()
    correct = 0
    precision = 0.0
    recall = 0.0
    f_score = 0.0
    for i in range(conf_matrix.shape[0]):
        tp = conf_matrix[i, i]
        if tp > 0:
            tp_fp = conf_matrix.sum(axis=0)[i]
            tp_fn = conf_matrix.sum(axis=1)[i]
            correct += tp
            precision += float(tp) / tp_fp
            recall += float(tp) / tp_fn
            f_score += float(2 * tp) / (tp_fp + tp_fn)
    accuracy = float(correct) / total_count
    precision /= conf_matrix.shape[0]
    recall /= conf_matrix.shape[0]
    f_score /= conf_matrix.shape[0]
    return accuracy, precision, recall, f_score    


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
    
    features = [list(range(num_feature))] * num_clf
    cluster = Cluster(num_clf, ['dt'], features, label_map, random_state=random_state)
    cluster.train(train_clf)
    mv_cmatrix = cluster.majorityVote(test)
    wv_cmatrix = cluster.weightedVote(test)

    real_set = [train_mdp.iloc[:, -1], test.iloc[:, -1]]
    res_set = [cluster.results(train_mdp), cluster.results(test)]
    prob_set = [cluster.resProb(train_mdp), cluster.resProb(test)]

    env = Environment(num_clf, real_set, res_set, prob_set, label_map, sequential=sequential)
    learn = get_learn_function(algorithm)
    model = learn(env, 0, num_training, learning_rate, epsilon, discount_factor, random_state, **network_kwargs)
    rl_cmatrix = env.evaluation(model, 1)

    print(mv_cmatrix)
    print(wv_cmatrix)
    print(rl_cmatrix)
    print(computeConfMatrix(mv_cmatrix))
    print(computeConfMatrix(wv_cmatrix))
    print(computeConfMatrix(rl_cmatrix))


if __name__ == '__main__':
    options = readCommand(sys.argv[1:])
    train(**options)


