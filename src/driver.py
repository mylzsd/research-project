import classifier
import mdp
import reader
import sys
import numpy as np
import random as rd


def readCommand(argv):
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-d", "--dataset", help = "experiment with DATASET", metavar = "DATASET")
    parser.add_option("-m", "--model", help = "use MODEL on reinforcement learning", metavar = "MODEL")
    parser.add_option("-n", "--num-clf", type = "int", help = "number of classifier, default value 50", default = 50)
    parser.add_option("-l", "--learning-rate", type = "float", help = "default value 0.1", default = 0.1)
    parser.add_option("-f", "--discount-factor", type = "float", help = "default value 1.0", default = 1.0)
    parser.add_option("-t", "--num-training", type = "int", help = "default value 10000", default = 10000)
    parser.add_option("-e", "--epsilon", type = "float", help = "default value 0.1", default = 0.1)
    parser.add_option("-r", "--random-state", type = "int", default = rd.randint(1, 10000))
    parser.add_option("-s", "--num-estimators", type = "int", default = 100)
    
    options, _ = parser.parse_args(argv)
    args = dict()
    args["dataset"] = options.dataset
    args["model"] = options.model
    args["num_clf"] = options.num_clf
    args["num_training"] = options.num_training
    args["learning_rate"] = options.learning_rate
    args["discount_factor"] = options.discount_factor
    args["epsilon"] = options.epsilon
    args["random_state"] = options.random_state
    args["n_estimators"] = options.num_estimators
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
    rdr = reader.Reader(kwargs["random_state"])
    train, train_clf, train_mdp, test = rdr.read(dataset)
    print(train_clf.shape)
    print(train_mdp.shape)
    print(test.shape)
    num_feature = train_clf.shape[1] - 1

    features = [list(range(num_feature))] * num_clf
    pure_clf = classifier.Cluster(num_clf, ["rf"], features, **kwargs)
    pure_clf.train(train)
    cluster = classifier.Cluster(num_clf, ["rf"], features, **kwargs)
    cluster.train(train_clf)

    # rl = mdp.MDP(cluster, model, learning_rate, discount_factor, epsilon, **kwargs)
    # rl.train(train_mdp, num_training, test)

    pure_scores = pure_clf.accuracy(test)
    clf_to_mdp = cluster.accuracy(train_mdp)
    clf_to_test = cluster.accuracy(test)
    # mdp_score = rl.accuracy(test)
    print(pure_scores)
    print(np.mean(pure_scores))
    print(clf_to_mdp)
    print(np.mean(clf_to_mdp))
    print(clf_to_test)
    print(np.mean(clf_to_test))
    # print(mdp_score)

    print(pure_clf.majorityVote(test))
    print(cluster.majorityVote(train_mdp))
    print(cluster.majorityVote(test))


if __name__ == "__main__":
    options = readCommand(sys.argv[1:])
    run(**options)

