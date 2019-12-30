import unittest

import numpy as np

from tests import TestEnv

from pyreinforce.brain import Brain
from pyreinforce.acting import EpsGreedyPolicy
from pyreinforce.memory import Memory
from pyreinforce import DdpgAgent


class TestBrain(Brain):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        seed = 123
        self._np_random = np.random.RandomState()
        self._np_random.seed(seed)

    def predict_a(self, states, is_target=False, **kwargs):
        if states.ndim < 2:
            states = np.expand_dims(states, axis=0)

        batch_size = states.shape[0]

        return self._np_random.uniform(size=(batch_size, self._n_outputs))

    def predict_q(self, states, actions, is_target=False, **kwargs):
        if states.ndim < 2:
            states = np.expand_dims(states, axis=0)

        if actions.ndim < 2:
            actions = np.expand_dims(actions, axis=0)

        assert states.shape[0] == actions.shape[0],\
            'Batch size does not match. states: {}, actions: {}'.format(states.shape[0], actions.shape[0])

        batch_size = states.shape[0]

        return self._np_random.uniform(size=(batch_size, self._n_outputs))

    def train(self, states, actions, targets, **kwargs):
        assert states.shape[0] == actions.shape[0],\
            'Batch size does not match. states: {}, actions: {}'.format(states.shape[0], actions.shape[0])
        assert states.shape[0] == targets.shape[0],\
            'Batch size does not match. states: {}, targets: {}'.format(states.shape[0], targets.shape[0])
        assert states.shape[1] == self._n_inputs,\
            'States shape. expected: {}, actual: {}'.format(self._n_inputs, states.shape[1])
        assert targets.shape[1] == self._n_outputs,\
            'Targets shape. expected: {}, actual: {}'.format(self._n_outputs, targets.shape[1])

        global_step = kwargs['global_step']
        train_freq = kwargs['train_freq']

        assert global_step % train_freq == 0,\
            'Train frequency not honored. global_step: {}, train_freq: {}'.format(global_step, train_freq)


class DdpgAgentTest(unittest.TestCase):
    def setUp(self):
        self._n_episodes = 10
        self._env = TestEnv()
        self._n_states = 3
        self._n_actions = 7
        self._brain = TestBrain(self._n_states, self._n_actions)
        self._eps = 0.1
        self._acting = EpsGreedyPolicy(self._eps)
        self._capacity = 1000
        self._batch_size = 8
        self._replay_memory = Memory(self._capacity, self._batch_size)
        self._gamma = 0.99
        self._train_freq = 4

        self._agent = DdpgAgent(self._n_episodes, self._env, self._brain, self._acting,
                                self._replay_memory, self._gamma, train_freq=self._train_freq)

    def test_run(self):
        rewards, _ = self._agent.run()

        self.assertEqual(self._n_episodes, len(rewards))


if __name__ == '__main__':
    unittest.main()
