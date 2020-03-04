from classifier import Ensemble
from environment import Environment
from adaboost import AdaBoost
import fs
import reader as rdr
import util as U
import random as rd
import numpy as np
import argparse
import time
import sys
from collections import Counter
from importlib import import_module
from sklearn.model_selection import KFold


LOAD_CLF = True
LOAD_IBRL = False


def readCommand(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', metavar='DATASET', required=True)
    parser.add_argument('-a', '--algorithm', metavar='ALGORITHM', required=True)
    parser.add_argument('-n', '--num-clf', type=int, default=100)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.1)
    parser.add_argument('-f', '--discount-factor', type=float, default=1.0)
    parser.add_argument('-t', '--num-training', type=int, default=10000)
    parser.add_argument('-e', '--epsilon', type=float, default=0.1)
    parser.add_argument('-r', '--random-state', type=int, default=6666)
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
          num_clf=100,
          num_training=10000,
          learning_rate=0.1,
          discount_factor=1.0,
          epsilon=1.0,
          portion=0.56,
          sequential=True,
          **network_kwargs):
    rd.seed(random_state)
    np.random.seed(random_state)
    
    start_time = time.time()
    data = rdr.read(dataset)
    time_cost = time.time() - start_time
    print('reading data takes %.3f sec' % (time_cost))
    print('data shape:', data.shape)
    # shuffle dataset
    # data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)


    num_feature = data.shape[1] - 1
    label_map = dict()
    # label_map[None] = 'N'
    for l in data.iloc[:, -1]:
        if l not in label_map:
            label_map[l] = len(label_map)
    print('number of labels: %d' % (len(label_map)))

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

    feature_type = 1
    features = list()
    for i in range(num_clf):
        if feature_type == 1:
            features.append(list(range(num_feature)))
        elif feature_type == 2:
            features.append(rd.choices(list(range(num_feature)), 
                k=int(np.ceil(num_feature * 0.5))))
        elif feature_type == 3:
            # every 1/3 classifiers get 1/3 features
            size = int((num_feature - 1) / 3) + 1
            index = int((num_clf - 1) / 3) + 1
            if i < index:
                features.append(list(range(size)))
            elif i < 2 * index:
                features.append(list(range(size, 2 * size)))
            else:
                features.append(list(range(2 * size, num_feature)))
    # print(features)

    mv_stat = [0.0] * 4
    wv_stat = [0.0] * 4
    fs_stat = [0.0] * 4
    adab_stat = [0.0] * 4
    eprl_stat = [0.0] * 4
    ibrl_stat = [0.0] * 4
    time_costs = [0.0] * 7
    fs_size = 0.0
    eprl_size = 0.0
    ibrl_size = 0.0
    avg_full_test_accu = 0.0
    avg_part_test_accu = 0.0

    term = 10
    kf = KFold(n_splits=term, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(kf.split(data)):
        print('\nRunning iteration %d of 10 fold...' % (i + 1))
        out_model = []
        out_res = []
        out_time = []
        train = data.iloc[train_idx, :]
        test = data.iloc[test_idx, :]
        train_clf, train_ens = rdr.splitByPortion(train, portion, random_state)
        # print(train_clf.shape, train_ens.shape, test.shape)

        # train or load ensembles
        start_time = time.time()
        # full ensemble
        persistence = 'models/clfs/d{}n{:d}c{:d}f{:d}r{:d}/i{:d}full/'.format(
            dataset, num_clf, clf_type, feature_type, random_state, i)
        if LOAD_CLF:
            full_ensemble = Ensemble(num_clf, clf_types, features, label_map, 
                persistence=persistence)
        else:
            full_ensemble = Ensemble(num_clf, clf_types, features, label_map, 
                random_state=random_state)
            full_ensemble.train(train)
            full_ensemble.saveClf(persistence)
        # part ensemble
        persistence = 'models/clfs/d{}n{:d}c{:d}f{:d}r{:d}/i{:d}part/'.format(
            dataset, num_clf, clf_type, feature_type, random_state, i)
        if LOAD_CLF:
            part_ensemble = Ensemble(num_clf, clf_types, features, label_map, 
                persistence=persistence)
        else:
            part_ensemble = Ensemble(num_clf, clf_types, features, label_map, 
                random_state=random_state)
            part_ensemble.train(train_clf)
            part_ensemble.saveClf(persistence)
        time_cost = time.time() - start_time
        time_costs[0] += time_cost
        print('%s ensembles takes %.3f sec' % 
            ('loading' if LOAD_CLF else 'training', time_cost))

        # creat environment
        start_time = time.time()
        real_set = [train_ens.iloc[:, -1], test.iloc[:, -1]]
        res_set = [part_ensemble.results(train_ens), part_ensemble.results(test)]
        prob_set = [part_ensemble.resProb(train_ens), part_ensemble.resProb(test)]
        env = Environment(num_clf, real_set, res_set, prob_set, label_map)
        time_cost = time.time() - start_time
        time_costs[1] += time_cost
        print('creating environment takes %.3f sec' % (time_cost))
    
        # get the performance of basic classifiers
        # full ensemble
        full_test_accu = full_ensemble.accuracy(test)
        avg_full_test_accu += np.mean(full_test_accu)
        # part ensemble
        part_test_accu = part_ensemble.accuracy(test)
        avg_part_test_accu += np.mean(part_test_accu)
        '''
        # voting techniques
        start_time = time.time()
        # majority vote
        mv_cmatrix = full_ensemble.majorityVote(test)
        # print(mv_cmatrix)
        mv_res = U.computeConfMatrix(mv_cmatrix)
        for s in range(4):
            mv_stat[s] += mv_res[s]
        out_model.append('mv')
        out_res.append(mv_res)
        # weighted vote
        wv_cmatrix = full_ensemble.weightedVote(test)
        # print(wv_cmatrix)
        wv_res = U.computeConfMatrix(wv_cmatrix)
        for s in range(4):
            wv_stat[s] += wv_res[s]
        out_model.append('wv')
        out_res.append(wv_res)
        time_cost = time.time() - start_time
        out_time.append(time_cost)
        time_costs[2] += time_cost
        print('voting takes %.3f sec' % (time_cost))
        
        # FS
        start_time = time.time()
        fs_model = fs.train(num_clf, real_set[0], res_set[0])
        fs_size += len(fs_model)
        fs_cmatrix = fs.evaluation(fs_model, real_set[1], res_set[1], label_map)
        # print(fs_cmatrix)
        fs_res = U.computeConfMatrix(fs_cmatrix)
        for s in range(4):
            fs_stat[s] += fs_res[s]
        out_model.append('fs')
        out_res.append(fs_res)
        time_cost = time.time() - start_time
        out_time.append(time_cost)
        time_costs[3] += time_cost
        print('FS takes %.3f sec' % (time_cost))
        
        # AdaBoost
        start_time = time.time()
        adab = AdaBoost(num_clf, random_state)
        adab.train(train)
        adab_cmatrix = adab.evaluation(test, label_map)
        adab_res = U.computeConfMatrix(adab_cmatrix)
        for s in range(4):
            adab_stat[s] += adab_res[s]
        out_model.append('adab')
        out_res.append(adab_res)
        time_cost = time.time() - start_time
        out_time.append(time_cost)
        time_costs[4] += time_cost
        print('AdaBoost takes %.3f sec' % (time_cost))
        
        # EPRL
        start_time = time.time()

        time_cost = time.time() - start_time
        out_time.append(time_cost)
        time_costs[5] += time_cost
        print('EPRL takes %.3f sec' % (time_cost))
        
        # IBRL
        start_time = time.time()
        model_folder = 'models/ibrls/d{}n{:d}c{:d}f{:d}r{:d}/'.format(
            dataset, num_clf, clf_type, feature_type, random_state)
        if not LOAD_IBRL and not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        model_path = '{}/i{:d}.ibrl'.format(model_folder, i)
        # print(model_path)
        if LOAD_IBRL:
            model.load(model_path)
        else:
            learn = get_learn_function(algorithm)
            model = learn(env, 0, num_training, learning_rate, epsilon, 
                discount_factor, random_state, **network_kwargs)
            model.save(model_path)
        ibrl_cmatrix, avg_clf = env.evaluation(model, 1, verbose=False)
        ibrl_size += avg_clf
        # print(ibrl_cmatrix)
        ibrl_res = U.computeConfMatrix(ibrl_cmatrix)
        for s in range(4):
            ibrl_stat[s] += ibrl_res[s]
        out_model.append('rl')
        out_res.append(ibrl_res)
        time_cost = time.time() - start_time
        out_time.append(time_cost)
        time_costs[6] += time_cost
        print('IBRL takes %.3f sec' % (time_cost))
        '''
        U.outputs(out_model, out_res)
        print(np.mean(full_test_accu))
        print(U.formatFloats(full_test_accu, 2) + '\n')
        print(np.mean(part_test_accu))
        print(U.formatFloats(part_test_accu, 2) + '\n')

    mv_stat = [n / term for n in mv_stat]
    wv_stat = [n / term for n in wv_stat]
    fs_stat = [n / term for n in fs_stat]
    adab_stat = [n / term for n in adab_stat]
    eprl_stat = [n / term for n in eprl_stat]
    ibrl_stat = [n / term for n in ibrl_stat]
    time_costs = [n / term for n in time_costs]
    fs_size /= term
    eprl_size /= term
    ibrl_size /= term
    avg_full_test_accu /= term
    avg_part_test_accu /= term
    U.outputs(['mv', 'wv', 'fs', 'adab', 'eprl', 'ibrl'], 
              [mv_stat, wv_stat, fs_stat, adab_stat, eprl_stat, ibrl_stat])
    print('time costs: C, E, V, FS, Ada, EPRL, IBRL\n       ' 
        + U.formatFloats(time_costs, 2))
    print('FS size: %.5f, EPRL size: %.5f, IBRL size: %.5f' 
        % (fs_size, eprl_size, ibrl_size))
    print('full test avg accu: %.5f, part test avg accu: %.5f' 
        % (avg_full_test_accu, avg_part_test_accu))


if __name__ == '__main__':
    options = readCommand(sys.argv[1:])
    train(**options)


