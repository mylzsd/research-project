import classifier
import random
import pandas as pd

FEATURE_COST = 0
LEARNING_RATE = 0.1
DISCOUNT = 1.0
CORRECT_REWARD = 1
EPSILON = 0.1

class TreeState:

	def __init__(self, size, value = None, is_term = False):
		self.size = size
		if value == None:
			self.value = [None] * size
		else:
			self.value = value[:]
		self.is_term = is_term

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
			if v is None: continue
			ret.add(i)
		return ret

	def feature(self, cluster):
		ret = set()
		for i, v in enumerate(self.value):
			if v is None: continue
			ret.update(cluster.features[i])
		return ret

	def __eq__(self, other):
		if not isinstance(other, TreeState): 
			return False
		return self.value == other.value and self.is_term == other.is_term

	def __hash__(self):
		t = tuple(self.value + [self.is_term])
		# print(t)
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
			# print("i is %d" % i)
			# print("prediction is %s" % prediction[i])
			if isinstance(prediction, pd.Series):
				state_p.setValueAt(i, prediction.iloc[i])
			else:
				state_p.setValueAt(i, prediction)
		return state_p

	def __str__(self):
		if self.is_term:
			return "Stop"
		return "Visit %d" % (self.index)


class MDP:

	def __init__(self, cluster):
		self.cluster = cluster
		self.policy = dict()
		self.q_table = dict()
		self.validation_count = dict()

	def getAction(self, state):
		return self.policy.get(state, None)

	def getQ(self, state, action):
		return self.q_table.get(state, 0)

	def getLegalActions(self, state):
		if state.is_term: return []
		value = state.getValue()
		actions = [Action(-1)]
		for i, v in enumerate(value):
			if v is None:
				actions.append(Action(i))
		return actions

	def getSuccessors(self, state):
		pass

	def updateDict(self, new_policy, new_q_table):
		for k, v in new_policy.items():
			self.policy[k] = v
		for k, v in new_q_table.items():
			self.q_table[k] = v

	def train(self, data):
		real = data.iloc[:, -1].reset_index(drop = True)
		results = self.cluster.results(data)
		predictions = pd.concat([results, real], axis = 1)

		# label_map is used in neural network
		self.label_map = dict()
		index = 0
		for label in real:
			if label in label_map: continue
			label_map[label] = index
			index += 1

		"""
		qLearning
			for each episode:
				shuffle records
				for each instance:
					from init state
					select next state by eps greedy
					if non-term state:
						get new Q value
						back prop Q-network
					else:
						get reward if correct??
				until dqn converge
		"""


	def qLearning(self, predictions, epoch):
		n = len(predictions.index)
		for i in range(epoch):
			for j in range(n):
				next_policy = dict()
				next_q_table = dict()
				init_ts = TreeState(self.cluster.size)
				self.qHelper(init_ts, predictions.iloc[j], next_policy, next_q_table)
				# monitor updates after 1 instance
				self.updateDict(next_policy, next_q_table)
			# monitor updates after 1 epoch
			if (i + 1) % 1 == 0:
				print("Epoch %d" % (i))
				for state in self.q_table.keys():
					q = self.getQ(state)
					a = self.getAction(state)
					# if self.getQ(state) > 0 and not a is None:
					# print("%s\npolicy: %s, Q: %f" % (state, a, q))

	def qHelper(self, state, prediction, next_policy, next_q_table):
		curr_q = self.getQ(state)
		# print("\t%s: %f" % (str(state), curr_q))
		if state.is_term:
			# currently using fixed weighted voting
			# TODO: change to a trainable network
			real = prediction.iloc[-1]
			res = self.decisionFunction(state)
			v_count = self.validation_count.get(state, 0)
			if res is None or res != real:
				next_q = (curr_q * v_count - CORRECT_REWARD) / (v_count + 1)
				# next_q = curr_q - CORRECT_REWARD
			else:
				next_q = (curr_q * v_count + CORRECT_REWARD) / (v_count + 1)
				# next_q = curr_q + CORRECT_REWARD
			self.validation_count[state] = v_count + 1
			# if next_q > 0:
			# print("\t\t%s: %f" % (state, next_q))
			next_q_table[state] = next_q
		else:
			# apply the best action
			# TODO: use e-greedy
			actions = self.getLegalActions(state)
			best_pairs = []
			best_q = -9999
			for a in actions:
				state_p = Action.applyAction(state, a, prediction)
				cost = len(state_p.feature(self.cluster) - state.feature(self.cluster)) * FEATURE_COST
				q = DISCOUNT * self.getQ(state_p) - cost
				# print(state_p, q)
				if q > best_q:
					best_q = q
					best_pairs = [(a, state_p)]
				elif q == best_q:
					best_pairs.append((a, state_p))
			# print("\t\t%f" % best_q)
			action, state_p = random.choice(best_pairs)
			next_q = curr_q + LEARNING_RATE * (best_q - curr_q)
			next_policy[state] = action
			next_q_table[state] = next_q
			self.qHelper(state_p, prediction, next_policy, next_q_table)

	def decisionFunction(self, state):
		value = state.getValue()
		vote = dict()
		for i, v in enumerate(value):
			if v is None: continue
			size = len(self.cluster.features[i])
			curr_v = vote.get(v, 0)
			vote[v] = curr_v + size
		pairs = [(v, k) for k, v in vote.items()]
		if len(pairs) > 0:
			_, res = max(pairs)
		else:
			res = None
		return res

	def majorityVote(self, state):
		value = state.getValue()
		vote = dict()
		for i, v in enumerate(value):
			pass
		pass

	def valueIteration(self):
		pass

	def policyIteration(self):
		pass

	def load(self, filename):
		pass

	def write(self, filename):
		pass

	def validation(self, data):
		total = len(data.index)
		correct = 0
		feature = 0
		predictions = self.cluster.results(data)
		for i in range(total):
			state = TreeState(self.cluster.size)
			while not state.is_term:
				action = self.getAction(state)
				if action is None:
					# print("No policy: %s" % (state))
					action = Action(-1)
				state_p = Action.applyAction(state, action, predictions.iloc[i])
				state = state_p
			res = self.decisionFunction(state)
			# get real result
			# get predicted result
			# modify counter
			real = data.iloc[i, -1]
			if res == real:
				correct += 1
			feature += len(state.feature(self.cluster))
		accuracy = correct / total
		cost = feature / total
		return accuracy, cost

