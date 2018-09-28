from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import tree
from PIL import Image
import numpy as np
import pandas as pd
import random as rd
import csv

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

def splitByPortion(data, portion, rd = 666):
	part1 = data.sample(frac = portion, random_state = rd)
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


