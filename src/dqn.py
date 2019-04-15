

class DQN:
	def __init__(self, env):
		self.env = env

	def policy(self, state, randomness=0.0):
		pass

	def qValue(self, state):
		pass

	def train(self):
		pass


def learn(env, in_set, num_training, learning_rate, epsilon, discount_factor, random_state):
	pass