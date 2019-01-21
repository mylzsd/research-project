import pandas as pd


"""
portion is for the first half
"""
def splitByPortion(data, portion, seed = 666):
	part1 = data.sample(frac = portion, random_state = seed)
	part2 = data.loc[~data.index.isin(part1.index), :]
	return (part1, part2)


def audiology():
	pass


def breast-cancer():
	pass


def breast-w():
	pass


def cmc():
	pass


def dematology():
	pass


def ecoli():
	pass


def glass():
	pass


def hepatitis():
	pass


def iris():
	iris = pd.read_csv("data/iris/iris.csv", header = None)
	bezd = pd.read_csv("data/iris/bezdekIris.csv", header = None)
	iris = pd.concat([iris, bezd], ignore_index = True)

	# iris_train_clf, iris_rest = splitByPortion(iris, 0.4)
	# iris_train_mdp, iris_test = splitByPortion(iris_rest, 0.8)
	# print(iris_train_clf.shape)
	# print(iris_train_mdp.shape)
	# print(iris_test.shape)
	# return (iris_train_clf, iris_train_mdp, iris_test)
	train, test = splitByPortion(iris, 0.8)
	return (train, train, test)


def lymphography():
	pass

