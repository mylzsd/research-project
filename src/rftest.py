from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from PIL import Image
import numpy as np
import pandas as pd
import random as rd
import csv

import classifier
import time

'''
train_attr = []
train_label = []

with open('adult.mdata.csv', newline = '') as f:
    reader = csv.reader(f)
    for row in reader:
    	attr = list(map(int, row[1:8]))
    	train_attr.append(attr)
    	label = int(row[8])
    	train_label.append(label)

test_attr = []
test_label = []

with open('adult.mtest.csv', newline = '') as f:
    reader = csv.reader(f)
    for row in reader:
    	attr = list(map(int, row[1:8]))
    	test_attr.append(attr)
    	label = int(row[8])
    	test_label.append(label)
'''

def splitByPortion(data, portion, seed = 666):
	part1 = data.sample(frac = portion, random_state = seed)
	part2 = data.loc[~data.index.isin(part1.index), :]
	return (part1, part2)

rd.seed(666)

card = pd.read_csv("data/creditcard/creditcard.csv")
card.drop("Time", axis = 1, inplace = True)

fraud = card.loc[card.Class == 1, :]
nonfraud = card.loc[card.Class == 0, :]

# sample = rd.sample(range(nonfraud.shape[0]), 500)
# newnf = nonfraud.iloc[sample, :]
# newcard = pd.concat([fraud, newnf])
# card = newcard
# print(card.shape)
# train, test = splitByPortion(card, 0.75)

f_train, f_test = splitByPortion(fraud, 0.5)
nf_train, nf_test = splitByPortion(nonfraud, 0.5)
train = pd.concat([f_train, nf_train])
# sample = rd.sample(range(nf_test.shape[0]), 250)
# nf_test = nf_test.iloc[sample, :]
# test = pd.concat([f_test, nf_test])

feature_count = [1, 5, 15, 29]
sample_count = [100, 250, 500, 1000, 2500, 10000]
scores = [[], [], [], []]
t = tree.DecisionTreeClassifier()
for i in range(4):
	feature_score = [[], [], [], [], [], []]
	for j in range(29):
		print("i: %d, j: %d" % (i, j))
		rd.seed(i * j)
		feature = rd.sample(range(29), feature_count[i])
		X = train.iloc[:, feature]
		y = train.Class
		t.fit(X, y)
		# use different portion of fraud case in test data
		for k in range(6):
			sample = rd.sample(range(nf_test.shape[0]), sample_count[k])
			new_nf_test = nf_test.iloc[sample, :]
			test = pd.concat([f_test, new_nf_test])
			X_test = test.iloc[:, feature]
			y_test = test.Class
			feature_score[k].append(t.score(X_test, y_test))
		# X_test = test.iloc[:, feature]
		# y_test = test.Class
		# scores[i].append(t.score(X_test, y_test))
	for k in range(6):
		scores[i].append(np.mean(feature_score[k]))

print(scores)
# print(np.mean(scores[0]))
# print(np.mean(scores[1]))
# print(np.mean(scores[2]))
# print(np.mean(scores[3]))


'''
rf = RandomForestClassifier(n_estimators = 30)
rf.fit(X, y)
print("Done training")
# print(rf.feature_importances_)
# print(rf.predict_proba(X_test))
print(rf.score(X_test, y_test))
'''



'''

# rf = RandomForestClassifier(n_estimators = 30)
# rf.fit(train_attr, train_label)

# print(rf.feature_importances_)

# # print(rf.predict_proba(test_attr))
# # print(rf.score(test_attr, test_label))
# res = rf.predict_proba(test_attr)

outfile = open('out.txt', 'w')
# for r in res:
# 	outfile.write("%s\n" % r)


rfs = [RandomForestClassifier(n_estimators = 30) for _ in range(50)]
for i in range(len(rfs)):
	print('building rf %d' % i)
	rfs[i].fit(train_attr, train_label)
print("Finished RF construction")

# accu = []

# for rf in rfs:
# 	accu.append(rf.score(test_attr, test_label))

# print(accu)

restb = []

for rf in rfs:
	res = rf.predict_proba(test_attr[:200])
	restb.append(res)

# print(restb[0][:15])

conftb = []

for i in range(len(restb[0])):
	conf = []
	for res in restb:
		c = int(res[i][test_label[i] - 1] * 100)
		# c = int(max(res[i][0], res[i][1]) * 100)
		# if (res[i][0] > res[i][1]) != (test_label[i] == 1):
		# 	c *= -1
		conf.append(c)
	conftb.append(conf)

for row in conftb:
	outfile.write("%s\n" % row)

# print(rfs[0].predict_proba([test_attr[0]]))

npa = np.asarray(conftb, np.uint8)
# print(npa)
im = Image.fromarray(npa).save('p1.png')
'''

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
		feature = rd.sample(range(num_feature), int(num_feature))
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
		feature = rd.sample(range(num_feature), int(num_feature / 2))
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


