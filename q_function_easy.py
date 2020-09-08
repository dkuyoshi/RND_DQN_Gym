import chainer
from chainer import Chain, Sequential, ChainList, Variable
from chainer import links as L
from chainer import functions as F

from chainerrl.action_value import DiscreteActionValue
from chainerrl.q_function import StateQFunction
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.links import FactorizedNoisyLinear


class DQNQFunctionEasy(Chain):
    def __init__(self, obs_size, n_actions, n_hidden=64, n_out=32):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden)
            self.l1 = L.Linear(n_hidden, n_actions)
            # self.l2 = L.Linear(n_out, n_actions)

    def __call__(self, x, ):
        h = F.relu(self.l0(x))
        h = self.l1(h)
        return DiscreteActionValue(h)
