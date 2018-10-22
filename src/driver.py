import classifier
import mdp
import time
import random
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

"""
portion is for the first half
"""
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

	# model = mdp.MDP(cluster)
	# model.qLearning(predictions, 50)
	# for state in model.q_table.keys():
	# 	q = model.getQ(state)
	# 	a = model.getAction(state)
	# 	# if self.getQ(state) > 0 and not a is None:
	# 	print("%s\npolicy: %s, Q: %f" % (state, a, q))
	# mdp_score, cost = model.validation(frog_test)
	# print(clf_scores)
	# print(mdp_score, cost)


def humanActivity():
	NUM_CLASSIFIER = 100

	ha_train = pd.read_csv("data/humanactivity/train.csv")
	ha_test = pd.read_csv("data/humanactivity/test.csv")
	ha_train.drop("subject", axis = 1, inplace = True)
	ha_test.drop("subject", axis = 1, inplace = True)
	ha_train_clf, ha_train_net = splitByPortion(ha_train, 0.5)
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
		feature = random.sample(range(num_feature), int(num_feature))
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


if __name__ == '__main__':
	# iris()
	for _ in range(10):
		frog()
	# for _ in range(10):
	# 	humanActivity()

