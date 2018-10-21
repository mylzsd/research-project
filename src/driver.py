import classifier
import mdp
import random
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier


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

	feature_size = [22, 22, 22, 22, 22, 22, 22, 22]
	features = []
	for i in range(len(feature_size)):
		feature = random.sample(range(num_feature), feature_size[i])
		features.append(feature)
	print(features)

	cluster = classifier.Cluster(8, ["rf"], features)
	cluster.train(frog_train_clf)
	results = cluster.results(frog_train_mdp)
	real = frog_train_mdp.iloc[:, -1].reset_index(drop = True)
	# print(real)
	predictions = pd.concat([results, real], axis = 1)
	# print(predictions)
	clf_scores = cluster.validation(frog_train_mdp)
	print(clf_scores)
	total = 0
	over_half = 0
	for i in range(len(predictions)):
		counter = 0
		row = predictions.iloc[i]
		for j in range(len(row)):
			if row.iloc[j] != row.iloc[-1]:
				counter += 1
		if counter > 0:
			total += 1
			print("%d, %d" % (i, counter))
			if counter <= 4:
				over_half += 1
	print(total)
	print(over_half)
	# predictions.to_csv("out/frog_prediction.txt", sep = "\t")
	return
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


def humanActivity():
	NUM_CLASSIFIER = 15

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
	# print(pure_clf_scores)

	features = []
	for i in range(NUM_CLASSIFIER):
		feature = random.sample(range(num_feature), int(num_feature / 10))
		features.append(feature)
	# features = [list(range(num_feature))] * NUM_CLASSIFIER
	# print(features)

	cluster = classifier.Cluster(NUM_CLASSIFIER, ["dt"], features)
	cluster.train(ha_train_clf)
	clf_scores = cluster.validation(ha_test)
	# print(clf_scores)
	# print(np.mean(clf_scores))

	activity_map = dict()
	index = 0
	for activity in ha_train.iloc[:, -1]:
		if activity in activity_map: continue
		activity_map[activity] = index
		index += 1
	# print(len(activity_map))
	# print(activity_map)

	real_train = ha_train_net.iloc[:, -1].reset_index(drop = True)
	results_train = cluster.results(ha_train_net)
	train_net_bi = []
	for i, row in results_train.iterrows():
		bi_array = [0] * len(activity_map) * NUM_CLASSIFIER
		for j, r in enumerate(row):
			index = j * len(activity_map) + activity_map[r]
			bi_array[index] = 1
		train_net_bi.append(bi_array)
	train_net_bi_df = pd.DataFrame(train_net_bi)
	train_net = pd.concat([train_net_bi_df, real_train], axis = 1)
	# print(train_net)

	real_test = ha_test.iloc[:, -1].reset_index(drop = True)
	results_test = cluster.results(ha_test)
	test_net_bi = []
	for i, row in results_test.iterrows():
		bi_array = [0] * len(activity_map) * NUM_CLASSIFIER
		for j, r in enumerate(row):
			index = j * len(activity_map) + activity_map[r]
			bi_array[index] = 1
		test_net_bi.append(bi_array)
	test_net_bi_df = pd.DataFrame(test_net_bi)
	test_net = pd.concat([test_net_bi_df, real_test], axis = 1)
	# print(test_net)

	input_size = int(len(activity_map) * NUM_CLASSIFIER)
	hidden_layers = [input_size, int(input_size / 2), int(input_size / 2), 15, 6]
	cnn = MLPClassifier(hidden_layer_sizes = tuple(hidden_layers), \
						alpha = 1e-5, activation = "relu", solver = "sgd", \
						random_state = 1, max_iter = 1000)
	X = train_net.iloc[:, list(range(len(activity_map) * NUM_CLASSIFIER))]
	y = train_net.iloc[:, -1]
	cnn.fit(X, y)
	X_test = test_net.iloc[:, list(range(len(activity_map) * NUM_CLASSIFIER))]
	y_test = test_net.iloc[:, -1]
	net_score = cnn.score(X_test, y_test)
	
	print(pure_clf_scores[0], np.mean(clf_scores), net_score)


if __name__ == '__main__':
	# iris()
	# frog()
	humanActivity()
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