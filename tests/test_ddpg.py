import unittest

import numpy as np

from tests import TestEnv

from pyreinforce.brain import Brain
from pyreinforce.acting import OrnsteinUhlenbeckPolicy
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

        self._weights = self._np_random.uniform(size=(n_inputs, n_outputs))

    def predict_a(self, states, is_target=False, **kwargs):
        if states.ndim < 2:
            states = np.expand_dims(states, axis=0)

        return np.matmul(states, self._weights)

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
        self._shape = self._n_actions
        self._mu = 0.7
        self._theta = 0.05
        self._sigma = 0.01
        self._acting = OrnsteinUhlenbeckPolicy(self._shape, self._mu,
                                               self._theta, self._sigma)
        self._capacity = 1000
        self._batch_size = 8
        self._replay_memory = Memory(self._capacity, self._batch_size)
        self._gamma = 0.99
        self._train_freq = 4

    def test_run_without_validation(self):
        validation_freq = None
        validation_episodes = None
        agent = DdpgAgent(self._n_episodes, self._env, self._brain, self._acting,
                          self._replay_memory, self._gamma, self._train_freq,
                          validation_freq, validation_episodes)
        rewards, _ = agent.run()

        self.assertEqual(self._n_episodes, len(rewards))

    def test_run_with_validation(self):
        validation_freq = 2
        validation_episodes = 10
        agent = DdpgAgent(self._n_episodes, self._env, self._brain, self._acting,
                          self._replay_memory, self._gamma, self._train_freq,
                          validation_freq, validation_episodes)
        rewards, _ = agent.run()
        rewards = np.array(rewards)

        n_validations = self._n_episodes // validation_freq
        self.assertSequenceEqual([n_validations, validation_episodes], rewards.shape)

    def test_act_training(self):
        agent = DdpgAgent(self._n_episodes, self._env, self._brain, self._acting,
                          self._replay_memory, self._gamma, self._train_freq)

        n_steps = 10000
        s = np.random.uniform(size=(n_steps, 3))
        raw_actions = agent._predict_a(s)
        noisy_actions = [agent._act(s[i], False, cur_step=i) for i in range(n_steps)]
        noise = noisy_actions - raw_actions

        noise_mean = noise.mean(axis=0)
        almost_equal = np.allclose(self._mu, noise_mean, atol=1e-2)
        self.assertTrue(almost_equal)

        noise_std = noise.std(axis=0)
        self.assertTrue(np.all(noise_std > 0))

    def test_act_validation(self):
        agent = DdpgAgent(self._n_episodes, self._env, self._brain, self._acting,
                          self._replay_memory, self._gamma, self._train_freq)

        n_steps = 10000
        s = np.random.uniform(size=(n_steps, 3))
        expected_actions = agent._predict_a(s)
        actual_actions = [agent._act(s[i], True) for i in range(n_steps)]

        almost_equal = np.allclose(expected_actions, actual_actions)
        self.assertTrue(almost_equal)


if __name__ == '__main__':
    unittest.main()
