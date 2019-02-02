import classifier
import random
import pandas as pd


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
		t = tuple(self.pred + [action])
		return hash(t)

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
		max_q = 0
		for a in actions:
			q = self.getQ(state, a)
			if q >= max_q:
				if q > max_q:
					candidates.clear()
				candidates.append(a)
		return random.choice(candidates)


	def getQ(self, state, action):
		s_a = state.getHash(action)
		return self.q_table.get(s_a, random.random())


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


	def train(self, data, num_training):
		real = data.iloc[:, -1].reset_index(drop = True)
		results = self.cluster.results(data)
		predictions = pd.concat([results, real], axis = 1)

		if self.model == "ql":
			self.qLearning(predictions, num_training)
		elif self.model == "sarsa":
			self.sarsa(predictions, num_training)
		elif self.model == "dqn":
			self.dqn()


	def qLearning(self, predictions, num_training):
		self.q_table = dict()
		n = len(predictions.index)
		for i in range(num_training):
			# shuffle incidents
			shuffled = predictions.sample(frac = 1)
			for j in range(n):
				self.qlHelper(shuffled.iloc[j])
			if (i + 1) % 100 == 0:
				print("Epoch %d" % (i + 1))
		print(len(self.q_table))


	def qlHelper(self, prediction):
		state = TreeState(self.cluster.size)
		while state != None:
			action = self.getAction(state, self.epsilon)
			state_p = self.applyAction(state, action, prediction)
			# compute factors for updating Q value
			s_a = state.getHash(action)
			q_sa = self.getQ(state, action)
			reward = 0
			q_sp = 0
			if action.index == -1:
				# compute reward
				real = prediction.iloc[-1]
				pred = self.majorityVote(state)
				if pred == real:
					reward = 1.0
			else:
				# compute max Q for next state
				actions = state_p.getLegalActions()
				for a in actions:
					q_sp = max(q_sp, self.getQ(state_p, a))
			self.q_table[s_a] = q_sa + self.learning_rate * (reward + self.discount_factor * q_sp - q_sa)
			# update current state
			state = state_p


	def sarsa(self, predictions, num_training):
		n = len(predictions.index)
		for i in range(num_training):
			# shuffle incidents
			shuffled = predictions.sample(frac = 1)
			for j in range(n):
				self.saHelper(shuffled.iloc[j])
			if (i + 1) % 100 == 0:
				print("Epoch %d" % (i + i))
		print(len(self.q_table))


	def saHelper(self, prediction):
		state = TreeState(self.cluster.size)
		action = self.getAction(state, self.epsilon)
		while state != None:
			state_p = self.applyAction(state, action, prediction)
			if state_p != None:
				action_p = self.getAction(state_p, self.epsilon)
			else:
				action_p = None
			# compute factors for updating Q value
			q_sp = 0
			reward = 0
			if action.index == -1:
				real = prediction.iloc[-1]
				pred = self.majorityVote(state)
				if pred == real:
					reward = 1.0
			else:
				q_sp = self.getQ(state_p, action_p)
			s_a = state.getHash(action)
			q_sa = self.getQ(state, action)
			self.q_table[s_a] = q_sa + self.learning_rate * (reward + self.discount_factor * q_sp - q_sa)
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
		for i, v in enumerate(value):
			if v is None: continue
			curr_v = vote.get(v, 0)
			vote[v] = curr_v + 1
		pairs = [(v, k) for k, v in vote.items()]
		if len(pairs) > 0:
			_, res = max(pairs)
		else:
			res = None
		return res


	def load(self, filename):
		pass


	def write(self, filename):
		pass


	def validation(self, data):
		total = len(data.index)
		correct = 0
		predictions = self.cluster.results(data)
		for i in range(total):
			# show path
			print()
			state = TreeState(self.cluster.size)
			# get predicted result
			while state != None:
				action = self.getAction(state, 0)
				# show path
				print(str(state) + " -> " + str(action))
				if action.index == -1:
					pred = self.majorityVote(state)
					break
				state_p = self.applyAction(state, action, predictions.iloc[i])
				state = state_p
			# get real result & modify counter
			real = data.iloc[i, -1]
			if pred == real:
				correct += 1
		accuracy = correct / total
		return accuracy



