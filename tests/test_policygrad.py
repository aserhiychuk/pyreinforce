import unittest

import numpy as np

from tests import TestEnv

from pyreinforce.brain import Brain
from pyreinforce.acting import SoftmaxPolicy
from pyreinforce import PolicyGradientAgent


def _softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class TestBrain(Brain):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        seed = 123
        self._np_random = np.random.RandomState()
        self._np_random.seed(seed)

    def predict_policy(self, states, **kwargs):
        if states.ndim < 2:
            states = np.expand_dims(states, axis=0)

        batch_size = states.shape[0]

        probs = [self._np_random.uniform(size=self._n_outputs) for _ in range(batch_size)]
        probs = [_softmax(prob) for prob in probs]
        probs = np.array(probs)

        return probs

    def train(self, states, actions, returns):
        assert states.shape[0] == actions.shape[0],\
            'Batch size does not match. states: {}, actions: {}'.format(states.shape[0], actions.shape[0])
        assert states.shape[0] == returns.shape[0],\
            'Batch size does not match. states: {}, returns: {}'.format(states.shape[0], returns.shape[0])
        assert states.shape[1] == self._n_inputs,\
            'States shape. expected: {}, actual: {}'.format(self._n_inputs, states.shape[1])


class PolicyGradientAgentTest(unittest.TestCase):
    def setUp(self):
        self._n_episodes = 10
        self._env = TestEnv()
        self._n_states = 3
        self._n_actions = 7
        self._brain = TestBrain(self._n_states, self._n_actions)
        self._acting = SoftmaxPolicy()
        self._gamma = 0.99
        self._agent = PolicyGradientAgent(self._n_episodes, self._env, self._brain,
                                          self._acting, self._gamma)

    def test_run(self):
        rewards, _ = self._agent.run()

        self.assertEqual(self._n_episodes, len(rewards))


if __name__ == '__main__':
    unittest.main()
