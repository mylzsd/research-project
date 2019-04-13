from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
import random as rd


class Classifier:

    def __init__(self, classifier_type, feature, **clf_kwarg):
        if classifier_type == 'dt':
            self.clf = DecisionTreeClassifier(**clf_kwarg)
        elif classifier_type == 'rf':
            self.clf = RandomForestClassifier(**clf_kwarg)
        elif classifier_type == 'svm':
            self.clf = SVC(**clf_kwarg)
        else:
            raise ValueError('unrecognized classifier type')

    def train(self, X, y):
        self.clf.fit(X, y)

    def accuracy(self, test_X, test_y):
        return self.clf.score(test_X, test_y)

    def result(self, X):
        return self.clf.predict(X)

    def resProb(self, X):
        return self.clf.predict_proba(X)

    def classes(self):
        return self.clf.classes_


class Cluster:

    def __init__(self, size, types, features, label_map, **clf_kwarg):
        if len(features) != size:
            raise ValueError('length of feature does not match number of classifiers')
        self.size = size
        self.features = features
        self.label_map = label_map
        self.clf_types = []
        self.classifiers = []
        for i in range(size):
            self.clf_types.append(types[i % len(types)])
            clf_kwarg['random_state'] = rd.randint(1, 10000)
            self.classifiers.append(Classifier(self.clf_types[i], features[i], **clf_kwarg))

    def train(self, data, bagging=None):
        for i in range(self.size):
            feature = self.features[i]
            # use specified features to form X
            X = data.iloc[:, feature]
            y = data.iloc[:, -1]
            self.classifiers[i].train(X, y)

    def accuracy(self, data):
        ret = []
        for i in range(self.size):
            feature = self.features[i]
            test_X = data.iloc[:, feature]
            test_y = data.iloc[:, -1]
            s = self.classifiers[i].accuracy(test_X, test_y)
            ret.append(s)
        return ret

    # return the matrix of result
    # each row for an instance
    # each column for a classifier
    def results(self, data):
        ret = []
        for i in range(self.size):
            X = data.iloc[:, self.features[i]]
            res = self.classifiers[i].result(X)
            ret.append(res)
        df = pd.DataFrame(ret)
        return df.T

    # return the matrix of probabilistic result
    # each element m_{i,j} is a possibility distribution
    # predicted by classifier j on instance i
    def resProb(self, data):
        ret = []
        for i in range(self.size):
            X = data.iloc[:, self.features[i]]
            prob = self.classifiers[i].resProb(X)
            # match the indices of classes in classifiers 
            # to the indices in the label map
            indices = [0] * len(self.label_map)
            for j, c in enumerate(self.classifiers[i].classes()):
                indices[self.label_map[c]] = j
            # reorder the probability distribution 
            # according to the indices in the label map
            prob = prob[:, indices]
            # reduce dimensionality
            tu_p = [tuple(p) for p in prob]
            ret.append(tuple(tu_p))
        df = pd.DataFrame(ret)
        return df.T

    # TODO: add f-score
    def majorityVote(self, data):
        count = 0
        for i in range(data.shape[0]):
            y = data.iloc[i, -1]
            vote = dict()
            for j in range(self.size):
                feature = self.features[j]
                X = data.iloc[i, feature].values.reshape(1, -1)
                pred = self.classifiers[j].result(X)[0]
                curr_v = vote.get(pred, 0)
                vote[pred] = curr_v + 1
            max_v = 0
            candidates = list()
            for k, v in vote.items():
                if v >= max_v:
                    if v > max_v:
                        candidates.clear()
                        max_v = v
                    candidates.append(k)
            if len(candidates) > 0 and rd.choice(candidates) == y:
                count += 1
        return float(count / data.shape[0])

    # TODO: implementation
    def weightedVote(self, data):
        pass


