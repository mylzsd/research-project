import random as rd


def majorityVote(state):
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
		return rd.choice(candidates)
	else:
		return None