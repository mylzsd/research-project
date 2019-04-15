from ql import Tabular
import random as rd


def learn(env, in_set, num_training, learning_rate, epsilon, discount_factor, random_state):
    model = Tabular(env)
    num_ins = env.numInstance(in_set)
    for i in range(1, num_training + 1):
        # randomly select instance
        in_row = rd.choice(list(range(num_ins)))
        state = env.initState()
        action = model.policy(state, randomness=epsilon)
        while state is not None:
            state_p, reward = env.step(state, action, in_set, in_row)
            action_p = model.policy(state_p, randomness=epsilon)
            # maybe modify model every k steps
            model.train(state, action, state_p, reward, learning_rate, discount_factor, action_p=action_p)
            state = state_p
            action = action_p
        # print some log indicates training progress
    return model
