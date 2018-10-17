import classifier
import mdp
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import random

NUM_CLASSIFIER = 10


def splitByPortion(data, portion, rd = 666):
	part1 = data.sample(frac = portion, random_state = rd)
	part2 = data.loc[~data.index.isin(part1.index), :]
	return (part1, part2)


def iris():
	iris = pd.read_csv("data/iris/iris.csv", header = None)
	bezd = pd.read_csv("data/iris/bezdekIris.csv", header = None)
	iris = pd.concat([iris, bezd], ignore_index = True)

	num_feature = iris.shape[1] - 1
	iris_train_clf, iris_rest = splitByPortion(iris, 0.4)
	iris_train_mdp, iris_test = splitByPortion(iris_rest, 0.8)
	print(iris_train_clf.shape)
	print(iris_train_mdp.shape)
	print(iris_test.shape)

	# feature_size = [1, 1, 2, 2, 3]
	# features = []
	# for i in range(len(feature_size)):
	# 	feature = random.sample(range(num_feature), feature_size[i])
	# 	features.append(feature)
	# print(features)
	features = [[0], [1], [2], [3], 
				[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], 
				[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], 
				[0, 1, 2, 3]]

	cluster = classifier.Cluster(15, ["dt"], features)
	cluster.train(iris_train_clf)
	results = cluster.results(iris_train_mdp)
	real = iris_train_mdp.iloc[:, -1].reset_index(drop = True)
	predictions = pd.concat([results, real], axis = 1)
	model = mdp.MDP(cluster)
	model.qLearning(predictions, 200)

	clf_scores = cluster.validation(iris_test)
	print(clf_scores)
	mdp_score, cost = model.validation(iris_test)
	print(mdp_score, cost)


def frog():
	frog = pd.read_csv("data/MFCCs/Frogs_MFCCs.csv")
	frog.drop("RecordID", axis = 1, inplace = True)
	frog.drop("Species", axis = 1, inplace = True)
	frog.drop("Genus", axis = 1, inplace = True)
	# frog.drop("Family", axis = 1, inplace = True)

	num_feature = frog.shape[1] - 1
	print(num_feature)
	
	frog_train_clf, frog_rest = splitByPortion(frog, 0.4)
	frog_train_mdp, frog_test = splitByPortion(frog_rest, 0.8)
	print(frog_train_clf.shape)
	print(frog_train_mdp.shape)
	print(frog_test.shape)

	feature_size = [3, 3, 3, 8, 8, 8, 10, 10]
	features = []
	for i in range(len(feature_size)):
		feature = random.sample(range(num_feature), feature_size[i])
		features.append(feature)
	print(features)
	cluster = classifier.Cluster(8, ["dt"], features)
	cluster.train(frog_train_clf)
	results = cluster.results(frog_train_mdp)
	real = frog_train_mdp.iloc[:, -1].reset_index(drop=True)
	# print(real)
	predictions = pd.concat([results, real], axis = 1)
	# print(predictions)

	model = mdp.MDP(cluster)
	model.qLearning(predictions, 50)
	for state in model.q_table.keys():
		q = model.getQ(state)
		a = model.getAction(state)
		# if self.getQ(state) > 0 and not a is None:
		print("%s\npolicy: %s, Q: %f" % (state, a, q))

	clf_scores = cluster.validation(frog_test)
	print(clf_scores)
	mdp_score, cost = model.validation(frog_test)
	print(mdp_score, cost)


if __name__ == '__main__':
	# iris()
	frog()
	# read data
	# partition data 4:4:2
	# set classifier count
	# set classifier type
	# set feature count for each classifier
	# train classifiers
	# initialize mdp
	# train mdp
	# test classifier cluster
	# test mdp