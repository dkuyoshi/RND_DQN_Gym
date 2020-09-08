import chainer
from chainer import Chain, Sequential, ChainList, Variable
from chainer import links as L
from chainer import functions as F

from chainerrl.action_value import DiscreteActionValue
from chainerrl.q_function import StateQFunction
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.links import FactorizedNoisyLinear
import numpy as np


class NN(Chain):
    def __init__(self, obs_size, n_hidden=128, n_out=64):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden)
            self.l1 = L.Linear(n_hidden, n_hidden)
            self.l2 = L.Linear(n_hidden, n_out)

    def __call__(self, x, ):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        h = self.l2(h)

        return h


class RNDModelEasy(object):
    def __init__(self, obs_size, n_hidden=64, n_out=32):
        self.target = NN(obs_size, n_hidden, n_out)
        self.predict = NN(obs_size, n_hidden, n_out)

    def get_instinct_reward(self, x):
        f_target = self.target(x)
        f_predict = self.predict(x)
        # L2ノルム
        instinct_reward = 10**-1 * np.sqrt(np.sum((f_predict.array - f_target.array)**2))
        return instinct_reward
