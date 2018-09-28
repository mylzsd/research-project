import classifier
import mdp

class T:

	def __init__(self, v):
		self.v = v[:]

	def __eq__(self, other):
		return self.v == other.v

if __name__ == '__main__':
	l = [1, 2, 3]
	l2 = [1, 4, 3]
	t1 = T(l)
	t2 = T(l2)
	print(l == l2)
	print(t1 == t2)
	pass
	# read data
	# partition data 4:4:2
	# set classifier count
	# set classifier type
	# set feature count for each classifier
	# train classifiers
	# initialize mdp
	# train mdp
	# test classifier cluster
	# test mdp