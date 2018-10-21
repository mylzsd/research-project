%matplotlib inline
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class MAB(ABC):
    
    @abstractmethod
    def play(self, tround, context):
    
    @abstractmethod
    def update(self, arm, reward, context):


class EpsGreedy(MAB):
    
    def __init__(self, narms, epsilon, Q0 = np.inf):
        # parameter check
        assert narms > 0, "number of arms should be positive"
        assert epsilon >= 0 and epsilon <= 1, "epsilon should between 0 and 1"
        self.narms = narms
        self.epsilon = epsilon
        self.Q0 = Q0
        # record the arm and reward history
        self.history = []
        
        
    def play(self, tround, context = None):
        # parameter check
        assert tround > 0 and tround <= len(self.history) + 1, "invalid tround"
        if np.random.random_sample() < self.epsilon:
            # exploration with probability of epsilon
            # random integer in range [0, narms)
            k = np.random.randint(self.narms)
            # make it [1, narms]
            return k + 1
        else:
            # exploitation with probability of 1 - epsilon
            reward = dict()
            count = dict()
            # sum up first tround - 1 records
            for i in range(tround - 1):
                a, r = self.history[i]
                reward[a] = reward.get(a, 0) + r
                count[a] = count.get(a, 0) + 1
            # find the Q value of each arm and put it into the 
            # candidate list if it is the current best value
            best_arms = []
            best_q = np.NINF
            for i in range(1, self.narms + 1):
                if i in count:
                    q = reward[i] / count[i]
                else:
                    q = self.Q0
                if q > best_q:
                    best_q = q
                    best_arms = [i]
                elif q == best_q:
                    best_arms.append(i)
            # randomly select from best candidates
            return np.random.choice(best_arms)
        
        
    def update(self, arm, reward, context = None):
        # parameter check
        assert arm > 0 and arm <= self.narms, "invalid arm index"
        # append new record
        self.history.append((arm, reward))
        

class UCB(MAB):
    
    def __init__(self, narms, rho, Q0 = np.inf):
        # parameter check
        assert narms > 0, "number of arms should be positive"
        assert rho > 0, "rho should be positive"
        self.narms = narms
        self.rho = rho
        self.Q0 = Q0
        # record the arm and reward history
        self.history = []
        
    
    def play(self, tround, context = None):
        # parameter check
        assert tround > 0 and tround <= len(self.history) + 1, "invalid tround"
        # sum up first tround - 1 records
        reward = dict()
        count = dict()
        for i in range(tround - 1):
            a, r = self.history[i]
            reward[a] = reward.get(a, 0) + r
            count[a] = count.get(a, 0) + 1
        # find the Q value of each arm and put it into the 
        # candidate list if it is the current best value
        best_arms = []
        best_q = np.NINF
        for i in range(1, self.narms + 1):
            if i in count:
                q = reward[i] / count[i] + np.sqrt(self.rho * np.log(tround) / count[i])
            else:
                q = self.Q0
            if q > best_q:
                best_q = q
                best_arms = [i]
            elif q == best_q:
                best_arms.append(i)
        # randomly select from best candidates
        return np.random.choice(best_arms)
        
        
    def update(self, arm, reward, context = None):
        # parameter check
        assert arm > 0 and arm <= self.narms, "invalid arm index"
        # append new record
        self.history.append((arm, reward))
        
    
def offlineEvaluate(mab, arms, rewards, contexts, nrounds = None):
    ret = []
    if not nrounds: return ret
    # index of event to be used
    index = 0
    for i in range(1, nrounds + 1):
        while True:
            # get the arm from policy for round i
            arm_p = mab.play(i, contexts[index])
            # break if the arm matched
            if arm_p == arms[index]: break
            # otherwise find next event
            index += 1
            if index >= len(arms): return ret
        # append the reward to return value
        ret.append(rewards[index])
        # update mab
        mab.update(arms[index], rewards[index], contexts[index])
        # increment index to avoid duplicated use
        index += 1
        if index >= len(arms):
            return ret
    return ret
    

# this section reads dataset.txt and construct
# arms, rewards, and contexts arrays
f = open("dataset.txt", "r")
lines = f.readlines()
arms = []
rewards = []
contexts = []
for l in lines:
    # split by space
    elements = l.split()
    # add different element to corresponded list
    arms.append(int(elements[0]))
    rewards.append(float(elements[1]))
    contexts.append([float(x) for x in elements[2:]])
    

mab = EpsGreedy(10, 0.05)
results_EpsGreedy = offlineEvaluate(mab, arms, rewards, contexts, 800)
print('EpsGreedy average reward', np.mean(results_EpsGreedy))


mab = UCB(10, 1.0)
results_UCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
print('UCB average reward', np.mean(results_UCB))


class LinUCB(MAB):
    
    def __init__(self, narms, ndims, alpha):
        # parameters check
        assert narms > 0 and ndims > 0 and alpha >= 0, "all parameters should be non-negative"
        self.narms = narms
        self.ndims = ndims
        self.alpha = alpha
        self.matrixA = []
        self.vectorb = []
        # initialize to identity matrix or zero vector
        mA0 = []
        vb0 = []
        for _ in range(narms):
            mA0.append(np.identity(ndims))
            vb0.append(np.asarray([[0.0] * ndims]).T)
        self.matrixA.append(mA0)
        self.vectorb.append(vb0)
        
        
    def play(self, tround, context):
        # parameter check
        assert tround > 0 and tround <= len(self.matrixA), "invalid tround"
        mAt = self.matrixA[tround - 1]
        vbt = self.vectorb[tround - 1]
        # find the Q value of each arm and put it into the 
        # candidate list if it is the current best value
        best_arms = []
        best_p = np.NINF
        for i in range(1, self.narms + 1):
            # read corresponding context and form column
            start = (i - 1) * self.ndims
            end = start + self.ndims
            ctx = np.asarray([context[start:end]]).T
            # compute p value of ith arm
            invA = inv(mAt[i - 1])
            theta = np.matmul(invA, vbt[i - 1])
            p = np.matmul(theta.T, ctx)[0][0] + self.alpha * np.sqrt(np.matmul(ctx.T, np.matmul(invA, ctx))[0][0])
            if p > best_p:
                best_p = p
                best_arms = [i]
            elif p == best_p:
                best_arms.append(i)
        # randomly select from best candidates
        return np.random.choice(best_arms)
        
    
    def update(self, arm, reward, context):
        # parameter check
        assert arm > 0 and arm <= self.narms, "invalid arm index"
        # get previous matrix and vector
        mA_old = self.matrixA[-1]
        vb_old = self.vectorb[-1]
        # read corresponding context and form column
        start = (arm - 1) * self.ndims
        end = start + self.ndims    
        ctx = np.asarray([context[start:end]]).T
        # update selected arm and copy others
        mA_new = []
        vb_new = []
        for i in range(1, self.narms + 1):
            mA_new.append(mA_old[i - 1].copy())
            vb_new.append(vb_old[i - 1].copy())
            if i == arm:
                mA_new[i - 1] += np.matmul(ctx, ctx.T)
                vb_new[i - 1] += reward * ctx
        self.matrixA.append(mA_new)
        self.vectorb.append(vb_new)
        
    
mab = LinUCB(10, 10, 1.0)
results_LinUCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
print('LinUCB average reward', np.mean(results_LinUCB))


x = range(1, len(results_EpsGreedy) + 1)
s_EpsGreedy = [0]
s_UCB = [0]
s_LinUCB = [0]
y_EpsGreedy = []
y_UCB = []
y_LinUCB = []
for i in range(1, len(results_EpsGreedy) + 1):
    # calculate the sum
    s_EpsGreedy.append(s_EpsGreedy[i - 1] + results_EpsGreedy[i - 1])
    s_UCB.append(s_UCB[i - 1] + results_UCB[i - 1])
    s_LinUCB.append(s_LinUCB[i - 1] + results_LinUCB[i - 1])
    # divide number of rounds
    y_EpsGreedy.append(s_EpsGreedy[i] / i)
    y_UCB.append(s_UCB[i] / i)
    y_LinUCB.append(s_LinUCB[i] / i)
plt.plot(x, y_EpsGreedy, label = "EpsGreedy")
plt.plot(x, y_UCB, label = "UCB")
plt.plot(x, y_LinUCB, label = "LinUCB")
plt.legend()
plt.show()


"""
Based on the experiment in the paper, 7 different 
alpha values are tested, then the average reward 
for each alpha value is printed and plot in graph.
The result shows that the highest mean reward is achieved
when alpha equals to 0.2.
"""
x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
y = []
for i in x:
    mab = LinUCB(10, 10, i)
    res = offlineEvaluate(mab, arms, rewards, contexts, 800)
    print("alpha: %.5f, mean reward: %.5f" % (i, np.mean(res)))
    y.append(np.mean(res))
plt.plot(x, y)
plt.show()


from sklearn.metrics.pairwise import rbf_kernel

class KernelUCB(MAB):
	
    def __init__(self, narms, ndims, gamma, eta, kern):
        # parameters check
        assert narms > 0 and ndims > 0 and gamma > 0 and eta > 0, "all parameters should be positive"
        self.narms = narms
        self.ndims = ndims
        self.gamma = gamma
        self.eta = eta
        self.kern = kern
        # add dummy item to match index with tround
        # vectors x
        self.ctx_history = [None]
        # vector y
        self.rewards = [None]
        # matrices K^-1
        self.k_inverse = [None]
        
    
    def play(self, tround, context):
        # parameter check
        assert tround > 0 and tround <= len(self.rewards), "invalid tround"
        if tround == 1:
            u = [1.0] + [0.0] * (self.narms - 1)
        else:
            u = []
            for i in range(1, self.narms + 1):
                # read corresponding context and form row
                s = (i - 1) * self.ndims
                e = s + self.ndims
                x_n = np.asarray([context[s:e]])
                # construct k_{x_n, t-1}
                kernels = []
                for x in self.ctx_history[1:tround]:
                    kernels.append(self.kern(x_n, x)[0][0])
                k_xnt_1 = np.asarray([kernels]).T
                # precomputation for k_{x_n, t-1}' X K_t-1
                kxK = np.matmul(k_xnt_1.T, self.k_inverse[tround - 1])
                v = self.kern(x_n, x_n)[0][0] - np.matmul(kxK, k_xnt_1)[0][0]
                # sometimes v is negative in test which is incapable for square root
                # sigma = np.sqrt(np.absolute(v))
                sigma = np.sqrt(v)
                # construct y_t-1
                y_t_1 = np.asarray([self.rewards[1:tround]]).T
                matrix_mult = np.matmul(kxK, y_t_1)
                u_nt = matrix_mult[0][0] + self.eta / np.sqrt(self.gamma) * sigma
                u.append(u_nt)
        # find the Q value of each arm and put it into the 
        # candidate list if it is the current best value
        best_arms = []
        best_q = np.NINF
        for i, q in enumerate(u):
            if q > best_q:
                best_q = q
                best_arms = [i + 1]
            elif q == best_q:
                best_arms.append(i + 1)
        # randomly select from best candidates
        return np.random.choice(best_arms)
        
    
    def update(self, arm, reward, context):
        # record context and corresponding reward
        start = (arm - 1) * self.ndims
        end = start+ self.ndims
        ctx = np.asarray([context[start:end]])
        # compute k(x_t, x_t)
        k_tt = self.kern(ctx, ctx)
        # construct inverse of matrix K
        if len(self.k_inverse) == 1:
            # when t = 1
            k_inv = inv(k_tt + self.gamma)
        else:
            # compute b = k_{x_t, t-1}
            kernels = []
            for x in self.ctx_history[1:]:
                kernels.append(self.kern(ctx, x)[0][0])
            b = np.asarray([kernels]).T
            # K_t-1'
            k_prev = self.k_inverse[-1]
            # precomputation for K_t-1' X b
            kxb = np.matmul(k_prev, b)
            # precomputation for b' X K_t-1'
            btxk = np.matmul(b.T, k_prev)
            # compute each component
            k_22 = inv(k_tt + self.gamma - np.matmul(b.T, kxb))
            k_11 = k_prev + k_22[0][0] * np.matmul(kxb, btxk)
            k_12 = -k_22[0][0] * kxb
            k_21 = -k_22[0][0] * btxk
            # concat matrices
            top = np.concatenate((k_11, k_12), axis = 1)
            bot = np.concatenate((k_21, k_22), axis = 1)
            k_inv = np.concatenate((top, bot))
        self.k_inverse.append(k_inv)
        self.ctx_history.append(ctx)
        self.rewards.append(reward)
        
    
# gamma = [0.5, 1.0, 1.5, 2.0]
# eta = [0.2, 0.5, 0.8, 1.0]
# for g in gamma:
#     for e in eta:
#         mab = KernelUCB(10, 10, g, e, rbf_kernel)
#         results_KernelUCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
#         print("gamma: %.1f, eta: %.1f, reward: %.5f" % (g, e, np.mean(results_KernelUCB)))
"""
gamma: 0.5, eta: 0.2, reward: 0.77000
gamma: 0.5, eta: 0.5, reward: 0.68625
gamma: 0.5, eta: 0.8, reward: 0.52375
gamma: 0.5, eta: 1.0, reward: 0.42250

gamma: 1.0, eta: 0.2, reward: 0.75375
gamma: 1.0, eta: 0.5, reward: 0.71125
gamma: 1.0, eta: 0.8, reward: 0.67000
gamma: 1.0, eta: 1.0, reward: 0.59375

gamma: 1.5, eta: 0.2, reward: 0.70875
gamma: 1.5, eta: 0.5, reward: 0.69625
gamma: 1.5, eta: 0.8, reward: 0.67625
gamma: 1.5, eta: 1.0, reward: 0.67875

gamma: 2.0, eta: 0.2, reward: 0.62750
gamma: 2.0, eta: 0.5, reward: 0.74750
gamma: 2.0, eta: 0.8, reward: 0.63125
gamma: 2.0, eta: 1.0, reward: 0.64250
Based on grid search result, although the performance is unstable,
gamma = 0.5 and eta = 0.2 is the best combination on average.
"""
mab = KernelUCB(10, 10, 0.5, 0.2, rbf_kernel)
results_KernelUCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
print("KernelUCB average reward ", np.mean(results_KernelUCB))
mab = LinUCB(10, 10, 0.2)
results_LinUCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
print("LinUCB average reward ", np.mean(results_LinUCB))

x = range(1, len(results_EpsGreedy) + 1)
y_LinUCB = [results_LinUCB[0]]
y_KernelUCB = [results_KernelUCB[0]]
for i in range(1, len(results_LinUCB)):
    # calculate sum of previous round
    s_LinUCB = y_LinUCB[i - 1] * i
    s_KernelUCB = y_KernelUCB[i - 1] * i
    # add current reward
    s_LinUCB += results_LinUCB[i]
    s_KernelUCB += results_KernelUCB[i]
    # divide number of rounds
    y_LinUCB.append(s_LinUCB / (i + 1))
    y_KernelUCB.append(s_KernelUCB / (i + 1))
plt.plot(x, y_LinUCB, label = "LinUCB")
plt.plot(x, y_KernelUCB, label = "KernelUCB")
plt.legend()
plt.show()
