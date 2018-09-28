import classifier
import pandas as pd

FEATURE_COST = 5
LEARNING_RATE = 0.1
DISCOUNT = 0.9

class TreeState:

	def __init__(self, size, value = None, is_term = False):
		self.size = size
		if value == None:
			self.value = [0] * size
		else:
			self.value = value[:]
		self.is_term = is_term
		self.valid_count = 0

	def setTerm(self, is_term):
		self.is_term = is_term

	def setValueAt(self, index, v):
		self.value[index] = v

	def setValue(self, value):
		if len(value) != self.size:
			raise ValueError("value length unmatched")
		self.value = value[:]

	def getValue(self):
		return self.value[:]

	def usedClassifier(self):
		ret = set()
		for i, v in enumerate(self.value):
			if v == 0: continue
			ret.add(i)
		return ret

	def feature(self, cluster):
		ret = set()
		for i, v in enumerate(self.value):
			if v == 0: continue
			ret.update(cluster.features[i])
		return ret

	def __eq__(self, other):
		if not isinstance(other, TreeState): 
			return False
		return self.value == other.value and self.is_term == other.is_term

	def __hash__(self):
		t = (tuple(self.value), self.is_term)
		return hash(t)

	def __str__(self):
		return " ".join(str(e) for e in self.value + [self.is_term])


class Action:

	def __init__(self, index):
		self.index = index
		self.is_term = index == -1
		
	def applyAction(state, action, prediction):
		state_p = TreeState(state.size, state.value)
		if action.is_term:
			state_p.setTerm(True)
		else:
			i = action.index
			state_p.setValueAt(i, prediction[i])
		return state_p


class MDP:

	def __init__(self, cluster):
		self.cluster = cluster
		self.policy = dict()
		self.q_table = dict()

	def getAction(self, state):
		return self.policy.get(state, 0)

	def getQ(self, state):
		return self.q_table.get(state, 0)

	def getLegalActions(self, state):
		if state.is_term: return []
		value = state.getValue()
		actions = [Action(-1)]
		for i, v in enumerate(value):
			if v == 0:
				actions.append(Action(i))
		return actions

	def getSuccessors(self, state):
		pass

	def qLearning(self, predictions):
		n = len(predictions.index)
		for i in range(EPOCH):
			next_policy = dict()
			next_q_table = dict()
			for j in range(n):
				init_ts = TreeState(self.cluster.size)
				qHelper(init_ts, predictions.iloc[j], next_policy, next_q_table)
				# monitor updates after 1 instance
			self.policy = next_policy
			self.q_table = next_q_table
			# monitor updates after 1 epoch

	def qHelper(state, prediction, next_policy, next_q_table):
		curr_q = self.getQ(state)
		if state.is_term:
			# currently using fixed weighted voting
			# TODO: change to a trainable network
			res = prediction.iloc[0][-1]
			total_size = 0
			total_value = 0
			value = state.getValue()
			for i, v in enumerate(value):
				if v == 0: continue
				size = len(self.cluster.features[i])
				total_size += size
				total_value += size * (1 if v == 1 else -1)
			total_q = state.valid_count * curr_q
			total_q += 100 if (total_value > 0) == (res == 0) else -100
			state.valid_count += 1
			next_q_table[state] = total_q / state.valid_count
		else:
			actions = self.getLegalActions(state)
			best_q = -9999
			for a in actions:
				state_p = Action.applyAction(state, a, prediction)
				cost = len(state_p.feature(self.cluster) - state.feature(self.cluster)) * FEATURE_COST
				temp_q = curr_q + LEARNING_RATE * (cost + DISCOUNT * self.getQ(state_p) - curr_q)
				if temp_q > best_q:
					best_q = temp_q
					best_action = a
				qHelper(state_p, prediction, next_policy, next_q_table)
			next_policy[state] = best_action
			next_q_table[state] = best_q

	def valueIteration(self):
		pass

	def policyIteration(self):
		pass

	def load(self, filename):
		pass

	def write(self, filename):
		pass

	def execute(self, case):
		pass
