class State:

	def __init__(self, pred=None, size=1, next_tree=0):
		if pred == None:
			self.pred = [None] * size
		else:
			self.pred = pred[:]
		self.next_tree = next_tree

	def getPred(self):
		return self.pred[:]

	def getLegalActions(self):
		if self.next_tree == len(self.pred):
			return ['eval']
		else:
			return ['visit', 'skip']

	def usedClassifier(self):
		ret = set()
		for i, v in enumerate(self.pred):
			if not v is None:
				ret.add(i)
		return ret

	def feature(self, cluster):
		ret = set()
		for i, v in enumerate(self.pred):
			if v is None: continue
			ret.update(cluster.features[i])
		return ret

	def oneHot(self, label_map):
		n = len(label_map)
		one_hot = [0] * len(self.pred) * n
		for i, v in enumerate(self.pred):
			if not v is None:
				one_hot[i * n + label_map[v]] = 1
		return one_hot

	def getHash(self, action):
		s = str(self) + str(action)
		return hash(s)

	def __eq__(self, other):
		if not isinstance(other, State): 
			return False
		return self.pred == other.pred

	def __hash__(self):
		t = tuple(self.pred)
		return hash(t)

	def __str__(self):
		return ' '.join(str(e) for e in self.pred) + ' next ' + str(self.next_tree)

