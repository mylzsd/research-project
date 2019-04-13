from collections import Counter
import random as rd


class State:

	def __init__(self, size, label_map):
		self.size = size
		self.label_map = label_map
		self.pred = [None] * size
		self.oh_pred = []
		for _ in range(size):
			self.oh_pred.append([0] * len(label_map))
		self.visited = set()

	def setPred(self, index, val):
		if index not in range(self.size):
			raise ValueError('index out of bound')
		visited.add(index)
		if val is not None:
			self.pred[index] = val
			self.oh_pred[index][self.label_map[val]] = 1

	def getPred(self, one_hot=False):
		if one_hot:
			ret = []
			for i in range(self.size):
				ret.append(self.oh_pred[i][:])
		else:
			ret = self.pred[:]
		return ret

	def usedClf(self):
		return self.visited.copy()

	def copy(self):
		ret = State(self.size, self.label_map)
		for i, v in enumerate(self.pred):
			if v is not None:
				ret.setPred(i, v)
		return ret

	def evaluation(self):
		c = Counter([p for p in self.pred if p is not None])
		m = max(c.values())
		return rd.choice([k for (k, v) in c.items() if v == m])

	def __str__(self):
		return ' '.join(str(e) for e in self.pred)


class Action:

	def __init__(self, index, visit=True):
		self.index = index
		self.visit = visit

	def __str__(self):
		if self.index == -1:
			return 'evaluation'
		else:
			return ('visit ' if self.visit else 'skip ') + str(self.index)


class Environment:
	# read in training and test results, including probabilistic results
	# whether act sequentially, label map, and feature related info
	def __init__(self,
				 num_clf,
				 sequential=True,
				 real_set,
				 res_set, 
				 prob_set, 
				 label_map, 
				 features=None, 
				 feature_cost=None):
		self.num_clf = num_clf
		self.sequential = sequential
		self.real_set = real_set
		self.res_set = res_set
		self.prob_set = prob_set
		self.label_map = label_map

	# return an initial blank state
	def initState(self):
		return State(self.num_clf, self.label_map)

	# perform state transition s, a -> s', r
	def step(self, state, action, in_set, in_row, deter=True):
		if deter:
			state_p = state.copy()
			reward = 0.0
			if action.index >= 0:
				# get result of target classifier and perform transition
				val = None
				if action.visit:
					val = self.res_set[in_set].iloc[in_row, action.index]
				state_p.setPred(action.index, val)
			else:
				# evaluation and get reward
				state_p = None
				pred = state.evaluation()
				if pred == self.real_set[in_set].iloc[in_row, 0]:
					reward = 1.0
		else:
			# TODO: nondeterministic state transition using probabilistic predictions
			pass
		return (state_p, reward)

	# get possible actions
	def legal_actions(self, state):
		if self.sequential:
			index = sorted(state.usedClf())[-1] + 1
			if index == self.num_clf:
				return [Action(-1)]
			else:
				return [Action(index), Action(index, visit=False)]
		else:
			# TODO: all unvisited classifier as well as evaluation
			pass

	# TODO: given a complete state return whether it predicts correctly

