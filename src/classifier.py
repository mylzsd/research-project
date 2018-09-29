from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd

class Classifier:

	def __init__(self, classifier_type, feature):
		self.feature = set(feature)
		if classifier_type == "dt":
			self.clf = DecisionTreeClassifier()
		elif classifier_type == "rf":
			self.clf = RandomForestClassifier(n_estimators = 30)
		elif classifier_type == "svm":
			self.clf = SVC()
		else:
			raise ValueError("unrecognized classifier type")

	def train(self, X, y):
		self.clf.fit(X, y)

	def validation(self, test_X, test_y):
		return self.clf.score(test_X, test_y)

	def result(self, X):
		return self.clf.predict(X)


class Cluster:

	def __init__(self, size, types, features):
		if len(features) != size:
			raise ValueError("length of feature does not match number of classifiers")
		self.size = size
		self.features = features
		self.classifiers = []
		self.clf_types = []
		for i in range(size):
			self.clf_types.append(types[i % len(types)])
			self.classifiers.append(Classifier(self.clf_types[i], features[i]))


	def train(self, data):
		for i in range(self.size):
			feature = self.features[i]
			# use specified features to form X
			X = data.iloc[:, feature]
			# select the last column as y
			y = data.iloc[:, -1]
			self.classifiers[i].train(X, y)

	def validation(self, data):
		ret = []
		for i in range(self.size):
			feature = self.features[i]
			test_X = data.iloc[:, feature]
			test_y = data.iloc[:, -1]
			s = self.classifiers[i].validation(test_X, test_y)
			ret.append(s)
		return ret

	def individualResult(self, X, index):
		return self.classifiers[index].result(X)

	def results(self, data):
		ret = []
		for i in range(self.size):
			feature = self.features[i]
			X = data.iloc[:, feature]
			res = self.classifiers[i].result(X)
			ret.append(res)
		df = pd.DataFrame(ret)
		return df.T

	def getTypes(self):
		return self.clf_types
