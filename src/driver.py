import classifier
import mdp
import reader
import time
import sys, random as rd
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

"""
portion is for the first half
"""
def splitByPortion(data, portion, seed = 666):
    part1 = data.sample(frac = portion, random_state = seed)
    part2 = data.loc[~data.index.isin(part1.index), :]
    return (part1, part2)


def iris():
    iris_train_clf, iris_train_mdp, iris_test = reader.iris()
    num_feature = iris_train_clf.shape[1] - 1

    # feature_size = [1, 1, 2, 2, 3]
    # features = []
    # for i in range(len(feature_size)):
    #   feature = random.sample(range(num_feature), feature_size[i])
    #   features.append(feature)
    # print(features)
    features = [[0], [1], [2], [3], 
                [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], 
                [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], 
                [0, 1, 2, 3]]

    cluster = classifier.Cluster(15, ["dt"], features)
    cluster.train(iris_train_clf)
    # results = cluster.results(iris_train_mdp)
    # real = iris_train_mdp.iloc[:, -1].reset_index(drop = True)
    # predictions = pd.concat([results, real], axis = 1)
    model = mdp.MDP(cluster)
    model.train(iris_train_mdp)
    # model.qLearning(predictions, 200)

    clf_scores = cluster.validation(iris_test)
    print(clf_scores)
    mdp_score = model.validation(iris_test)
    print(mdp_score)


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


def humanActivity():
    NUM_CLASSIFIER = 100

    ha_train = pd.read_csv("data/humanactivity/train.csv")
    ha_test = pd.read_csv("data/humanactivity/test.csv")
    ha_train.drop("subject", axis = 1, inplace = True)
    ha_test.drop("subject", axis = 1, inplace = True)
    # ha_train_clf, ha_train_net = splitByPortion(ha_train, 0.5)
    ha_train_clf = ha_train
    ha_train_net = ha_train
    # print(ha_train_clf.shape)
    # print(ha_train_net.shape)
    # print(ha_test.shape)
    num_feature = ha_train.shape[1] - 1

    pure_clf = classifier.Cluster(1, ["rf"], [list(range(num_feature))])
    pure_clf.train(ha_train)
    pure_clf_scores = pure_clf.validation(ha_test)
    # print(pure_clf_scores[0])

    features = []
    for i in range(NUM_CLASSIFIER):
        feature = random.sample(range(num_feature), int(num_feature / 2))
        features.append(feature)
    # print(features)

    cluster = classifier.Cluster(NUM_CLASSIFIER, ["rf"], features)
    cluster.train(ha_train_clf)
    clf_scores = cluster.validation(ha_test)
    # print(clf_scores)
    # print(np.mean(clf_scores))

    start_time = time.time()

    label_map = dict()
    index = 0
    for label in ha_train.iloc[:, -1]:
        if label in label_map: continue
        label_map[label] = index
        index += 1
    # print(len(label_map))
    # print(label_map)

    real_train = ha_train_net.iloc[:, -1].reset_index(drop = True)
    results_train = cluster.results(ha_train_net)
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

    real_test = ha_test.iloc[:, -1].reset_index(drop = True)
    results_test = cluster.results(ha_test)
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
    hidden_layers = [input_size, int(input_size / 2), int(input_size / 2), 15, 6]
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
    print("--- %s seconds ---" % (time.time() - start_time))


def test():
    train_clf, train_mdp, test = reader.lymphography()
    print(train_clf.shape)
    print(train_mdp.shape)
    print(test.shape)
    num_feature = train_clf.shape[1] - 1
    features = [list(range(num_feature))] * 100
    cluster = classifier.Cluster(100, ["rf"], features)
    cluster.train(train_clf)

    clf_scores = cluster.validation(test)
    print(clf_scores)
    print(np.mean(clf_scores))


def readCommand(argv):
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-d", "--dataset", help = "experiment with DATASET", metavar = "DATASET")
    parser.add_option("-m", "--model", help = "use MODEL on reinforcement learning", metavar = "MODEL")
    parser.add_option("-n", "--num-rf", type = "int", help = "number of random forest, default value 100", default = 100)
    parser.add_option("-l", "--learning-rate", type = "float", help = "default value 0.1", default = 0.1)
    parser.add_option("-f", "--discount-factor", type = "float", help = "default value 1.0", default = 1.0)
    parser.add_option("-t", "--num-training", type = "int", help = "default value 10000", default = 10000)
    parser.add_option("-e", "--epsilon", type = "float", help = "default value 0.1", default = 0.1)
    
    options, _ = parser.parse_args(argv)
    args = dict()
    args['dataset'] = options.dataset
    args['model'] = options.model
    args['num_rf'] = options.num_rf
    args['num_training'] = options.num_training
    args['learning_rate'] = options.learning_rate
    args['discount_factor'] = options.discount_factor
    args['epsilon'] = options.epsilon
    return args


def runExperiment(dataset, model, num_rf, num_training, learning_rate, discount_factor, epsilon):
    data_map = {
        "audiology": reader.audiology,
        "breast_cancer": reader.breast_cancer,
        "breast_w": reader.breast_w,
        "cmc": reader.cmc,
        "dematology": reader.dematology,
        "ecoli": reader.ecoli,
        "glass": reader.glass,
        "hepatitis": reader.hepatitis,
        "iris": reader.iris,
        "lymphography": reader.lymphography
    }
    train_clf, train_mdp, test = data_map[dataset]()
    print(train_clf.shape)
    print(train_mdp.shape)
    print(test.shape)
    num_feature = train_clf.shape[1] - 1

    features = [list(range(num_feature))] * num_rf
    cluster = classifier.Cluster(num_rf, ["rf"], features)
    cluster.train(train_clf)

    rl = mdp.MDP(cluster, model, learning_rate, discount_factor, epsilon)
    rl.train(train_mdp, num_training)

    clf_scores = cluster.validation(test)
    print(clf_scores)
    print(np.mean(clf_scores))
    mdp_score = rl.validation(test)
    print(mdp_score)


if __name__ == '__main__':
    options = readCommand(sys.argv[1:])
    runExperiment(**options)
    # test()

