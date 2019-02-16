import classifier
import random
import pandas as pd

debug_print = False

class TreeState:

	def __init__(self, size, pred = None):
		self.size = size
		if pred == None:
			self.pred = [None] * size
		else:
			self.pred = pred[:]

	def setPredAt(self, index, v):
		self.pred[index] = v

	def getPredAt(self, index):
		return self.pred[index]

	def setPred(self, pred):
		if len(pred) != self.size:
			raise ValueError("predictions length unmatched")
		self.pred = pred[:]

	def getPred(self):
		return self.pred[:]

	def getLegalActions(self):
		actions = [Action(-1)]
		for i, p in enumerate(self.pred):
			if p is None:
				actions.append(Action(i))
		return actions

	def usedClassifier(self):
		ret = set()
		for i, v in enumerate(self.pred):
			if v is None: continue
			ret.add(i)
		return ret

	def feature(self, cluster):
		ret = set()
		for i, v in enumerate(self.pred):
			if v is None: continue
			ret.update(cluster.features[i])
		return ret

	def getHash(self, action):
		s = str(self) + str(action)
		return hash(s)

	def __eq__(self, other):
		if not isinstance(other, TreeState): 
			return False
		return self.pred == other.pred

	def __hash__(self):
		t = tuple(self.pred)
		return hash(t)

	def __str__(self):
		return " ".join(str(e) for e in self.pred)


class Action:

	def __init__(self, index):
		self.index = index


	def __str__(self):
		if self.index == -1:
			return "Stop"
		return "Visit %d" % (self.index)


class MDP:

	def __init__(self, cluster, model, learning_rate, discount_factor, epsilon):
		self.cluster = cluster
		self.model = model
		self.learning_rate = learning_rate
		self.discount_factor = discount_factor
		self.epsilon = epsilon
		self.policy = dict()
		self.q_table = dict()


	def getAction(self, state, randomness):
		actions = state.getLegalActions()
		if random.random() < randomness:
			return random.choice(actions)
		candidates = list()
		max_q = float("-inf")
		for a in actions:
			q = self.getQ(state, a)
			if debug_print:
				print("[%s] (%f)" % (a, q))
			if q >= max_q:
				if q > max_q:
					candidates.clear()
					max_q = q
				candidates.append(a)
		return random.choice(candidates)


	def getQ(self, state, action):
		s_a = state.getHash(action)
		# return self.q_table.get(s_a, random.random())
		return self.q_table.get(s_a, 0)


	def applyAction(self, state, action, prediction):
		if action.index == -1:
			state_p = None
		else:
			state_p = TreeState(state.size, state.pred)
			i = action.index
			if isinstance(prediction, pd.Series):
				state_p.setPredAt(i, prediction.iloc[i])
			else:
				state_p.setPredAt(i, prediction)
		return state_p


	def train(self, data, num_training, test):
		real = data.iloc[:, -1].reset_index(drop = True)
		results = self.cluster.results(data)
		predictions = pd.concat([results, real], axis = 1)

		n = len(predictions.index)
		for i in range(num_training):
			# shuffle incidents
			shuffled = predictions.sample(frac = 1)
			for j in range(n):
				if self.model == "ql":
					self.qLearning(shuffled.iloc[j])
				elif self.model == "sarsa":
					self.sarsa(shuffled.iloc[j])
				elif self.model == "dqn":
					pass
			if (i + 1) % 1000 == 0:
				print("Episode %d: accuracy: %f" % (i + 1, self.validation(test)))
		print(len(self.q_table))


	def qLearning(self, prediction):
		state = TreeState(self.cluster.size)
		while state != None:
			if debug_print:
				print("\n{%s}" % (state))
			action = self.getAction(state, self.epsilon)
			state_p = self.applyAction(state, action, prediction)
			# compute factors for updating Q value
			s_a = state.getHash(action)
			q_sa = self.getQ(state, action)
			reward = 0
			if action.index == -1:
				q_sp = 0
				# compute reward
				real = prediction.iloc[-1]
				pred = self.majorityVote(state)
				if pred == real:
					reward = 1.0
				else:
					reward = -1.0
			else:
				# compute max Q for next state
				q_sp = float("-inf")
				actions = state_p.getLegalActions()
				for a in actions:
					q_sp = max(q_sp, self.getQ(state_p, a))
			self.q_table[s_a] = q_sa + self.learning_rate * (reward + self.discount_factor * q_sp - q_sa)
			if debug_print:
				print("[%s]->{%s}" % (str(action), str(state_p)))
				print("[%f] [%f] (%f)->(%f)\n" % (reward, q_sp, q_sa, self.q_table[s_a]))
			# update current state
			state = state_p


	def sarsa(self, prediction):
		state = TreeState(self.cluster.size)
		action = self.getAction(state, self.epsilon)
		while state != None:
			if debug_print:
				print("\n{%s}" % (state))
			state_p = self.applyAction(state, action, prediction)
			if state_p != None:
				action_p = self.getAction(state_p, self.epsilon)
			else:
				action_p = None
			# compute factors for updating Q value
			if action.index == -1:
				q_sp = 0
				real = prediction.iloc[-1]
				pred = self.majorityVote(state)
				if pred == real:
					reward = 1.0
				else:
					reward = -1.0
			else:
				q_sp = self.getQ(state_p, action_p)
				reward = 0
			s_a = state.getHash(action)
			q_sa = self.getQ(state, action)
			self.q_table[s_a] = q_sa + self.learning_rate * (reward + self.discount_factor * q_sp - q_sa)
			if debug_print:
				print("\n\n{%s}->[%s]->{%s}->[%s]" % (str(state), str(action), str(state_p), str(action_p)))
				print("[%f] [%f] (%f)->(%f)" % (reward, q_sp, q_sa, self.q_table[s_a]))
			# update current state and action
			state = state_p
			action = action_p


	def dqn(self, predictions, label_map):
		pass


	def featureVote(self, state):
		value = state.getPred()
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
		value = state.getPred()
		vote = dict()
		for v in value:
			if v is None: continue
			curr_v = vote.get(v, 0)
			vote[v] = curr_v + 1
		max_v = 0
		candidates = list()
		for k, v in vote.items():
			if v >= max_v:
				if v > max_v:
					candidates.clear()
					max_v = v
				candidates.append(k)
		if len(candidates) > 0:
			return random.choice(candidates)
		else:
			return None


	def load(self, filename):
		pass


	def write(self, filename):
		pass


	def validation(self, data):
		total = len(data.index)
		correct = 0
		predictions = self.cluster.results(data)
		for i in range(total):
			state = TreeState(self.cluster.size)
			# get predicted result
			while state != None:
				action = self.getAction(state, 0)
				# show path
				if debug_print:
					print("{%s}->[%s]" % (str(state), str(action)))
				if action.index == -1:
					pred = self.majorityVote(state)
					break
				state_p = self.applyAction(state, action, predictions.iloc[i])
				state = state_p
			# get real result & modify counter
			real = data.iloc[i, -1]
			if pred == real:
				correct += 1
			if debug_print:
				print("pred: %s, real: %s, correct: %s\n" % (pred, real, pred == real))
		accuracy = correct / total
		return accuracy



