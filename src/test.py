import classifier
import mdp
import reader
import time
import os, sys, random as rd
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from optparse import OptionParser

"""
portion is for the first half
"""
def splitByPortion(data, portion, seed = 666):
    part1 = data.sample(frac = portion, random_state = seed)
    part2 = data.loc[~data.index.isin(part1.index), :]
    return (part1, part2)


def frog():
    NUM_CLASSIFIER = 100

    frog = pd.read_csv("data/MFCCs/Frogs_MFCCs.csv")
    frog.drop("RecordID", axis = 1, inplace = True)
    frog.drop("Species", axis = 1, inplace = True)
    frog.drop("Genus", axis = 1, inplace = True)
    # frog.drop("Family", axis = 1, inplace = True)
    frog_train, frog_test = splitByPortion(frog, 0.8)
    frog_train_clf, frog_train_mdp = splitByPortion(frog_train, 0.5)
    # print(frog_train_clf.shape)
    # print(frog_train_mdp.shape)
    # print(frog_test.shape)
    num_feature = frog.shape[1] - 1

    time_1 = time.time()
    pure_clf = classifier.Cluster(1, ["rf"], [list(range(num_feature))])
    pure_clf.train(frog_train)
    pure_clf_scores = pure_clf.validation(frog_test)
    # print(pure_clf_scores[0])

    time_2 = time.time()
    features = []
    for i in range(NUM_CLASSIFIER):
        feature = random.sample(range(num_feature), int(num_feature))
        features.append(feature)
    # print(features)

    cluster = classifier.Cluster(NUM_CLASSIFIER, ["rf"], features)
    cluster.train(frog_train_clf)
    clf_scores = cluster.validation(frog_test)
    # print(clf_scores)
    # print(np.mean(clf_scores))

    label_map = dict()
    index = 0
    for label in frog_train.iloc[:, -1]:
        if label in label_map: continue
        label_map[label] = index
        index += 1
    # print(len(label_map))
    # print(label_map)

    real_train = frog_train.iloc[:, -1].reset_index(drop = True)
    results_train = cluster.results(frog_train)
    train_net_bi = []
    for i, row in results_train.iterrows():
        bi_array = [0] * len(label_map) * NUM_CLASSIFIER
        for j, r in enumerate(row):
            index = j * len(label_map) + label_map[r]
            bi_array[index] = 1
        train_net_bi.append(bi_array)
    train_net_bi_df = pd.DataFrame(train_net_bi)
    train_net = pd.concat([train_net_bi_df, real_train], axis = 1)
    # print(train_net)

    real_test = frog_test.iloc[:, -1].reset_index(drop = True)
    results_test = cluster.results(frog_test)
    test_net_bi = []
    for i, row in results_test.iterrows():
        bi_array = [0] * len(label_map) * NUM_CLASSIFIER
        for j, r in enumerate(row):
            index = j * len(label_map) + label_map[r]
            bi_array[index] = 1
        test_net_bi.append(bi_array)
    test_net_bi_df = pd.DataFrame(test_net_bi)
    test_net = pd.concat([test_net_bi_df, real_test], axis = 1)
    # print(test_net)

    input_size = int(len(label_map) * NUM_CLASSIFIER)
    hidden_layers = [input_size, int(input_size / 2), 8, 4]
    cnn = MLPClassifier(hidden_layer_sizes = tuple(hidden_layers), \
                        alpha = 1e-5, activation = "relu", solver = "sgd", \
                        random_state = 1, max_iter = 1000)
    X = train_net.iloc[:, list(range(len(label_map) * NUM_CLASSIFIER))]
    y = train_net.iloc[:, -1]
    cnn.fit(X, y)
    X_test = test_net.iloc[:, list(range(len(label_map) * NUM_CLASSIFIER))]
    y_test = test_net.iloc[:, -1]
    net_score = cnn.score(X_test, y_test)
    
    print(pure_clf_scores[0], np.mean(clf_scores), net_score)
    print("--- %s %s ---" % (time_2 - time_1, time.time() - time_2))


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


def mlp(dataset, num_clf):
    train, train_clf, train_net, test = reader.read(dataset)
    print(train_clf.shape)
    print(train_net.shape)
    print(test.shape)
    num_feature = train_clf.shape[1] - 1

    print("Training classifiers:", time.asctime(time.localtime(time.time())))
    pure_clf = classifier.Cluster(num_clf, ["rf"], [list(range(num_feature))] * num_clf)
    pure_clf.train(train)
    pure_scores = pure_clf.validation(test)

    features = []
    for i in range(num_clf):
        feature = rd.sample(range(num_feature), int(num_feature / 2))
        features.append(feature)

    cluster = classifier.Cluster(num_clf, ["rf"], features)
    cluster.train(train_clf)
    clf_scores = cluster.validation(test)

    print("Training mlp:", time.asctime(time.localtime(time.time())))

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
    print("Number of label:", len(label_map))

    train_bi = class2binary(cluster, num_clf, label_map, train_net)
    test_bi = class2binary(cluster, num_clf, label_map, test)

    input_size = int(len(label_map) * num_clf)
    hidden_layers = (input_size, int(input_size / 2), int(input_size / 2), len(label_map) * 2, len(label_map))
    mlp = MLPClassifier(hidden_layer_sizes = hidden_layers,
                        alpha = 1e-5, activation = "relu", solver = "sgd",
                        random_state = 1, max_iter = 10000)
    X = train_bi.iloc[:, list(range(len(label_map) * num_clf))]
    y = train_bi.iloc[:, -1]
    mlp.fit(X, y)
    X_test = test_bi.iloc[:, list(range(len(label_map) * num_clf))]
    y_test = test_bi.iloc[:, -1]
    net_score = mlp.score(X_test, y_test)
    
    print(min(pure_scores), max(pure_scores), np.mean(pure_scores))
    print(min(clf_scores), max(clf_scores), np.mean(clf_scores))
    print(net_score)
    print("Finish time:", time.asctime(time.localtime(time.time())))


def printResults(pred, real, label_map):
    for i in range(pred.shape[0]):
        same = True
        r = real.iloc[i, real.shape[1] - 1]
        p = list()
        for j in range(pred.shape[1]):
            p.append(label_map[pred.iloc[i, j]])
            if same and pred.iloc[i, j] != r:
                same = False
        if same:
            print("%s -> %s" % (p, label_map[r]))
        else:
            print("%s -> %s =====" % (p, label_map[r]))


def examine(dataset, num_clf, reuse):
    train, train_clf, train_mdp, test = reader.read(dataset)
    print(train_clf.shape)
    print(train_mdp.shape)
    print(test.shape)

    num_feature = train.shape[1] - 1
    features = [list(range(num_feature))] * num_clf
    cluster = classifier.Cluster(num_clf, ["rf"], features)
    
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
    print("Number of label:", len(label_map))

    if reuse:
        cluster.train(train)
        train_res = cluster.results(train)
        printResults(train_res, train, label_map)
    else:
        cluster.train(train_clf)
        train_res = cluster.results(train_mdp)
        printResults(train_res, train_mdp, label_map)
    print("\n")
    test_res = cluster.results(test)
    printResults(test_res, test, label_map)

    clf_scores = cluster.validation(test)
    print(clf_scores)
    print(np.mean(clf_scores))


def readCommand(argv):
    parser = OptionParser()
    parser.add_option("-m", "--mode", help = "examine dataset or run mlp test", metavar = "MODE")
    parser.add_option("-d", "--dataset", help = "experiment with DATASET", metavar = "DATASET")
    parser.add_option("-n", "--num-clf", type = "int", help = "number of classifier, default value 50", default = 50)
    parser.add_option("-r", action = "store_true", dest = "reuse", default = False)
    
    options, _ = parser.parse_args(argv)
    return options


if __name__ == "__main__":
    options = readCommand(sys.argv[1:])
    if options.mode == "examine":
        examine(options.dataset, options.num_clf, options.reuse)
    elif options.mode == "mlp":
        mlp(options.dataset, options.num_clf)

