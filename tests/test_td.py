import unittest

import numpy as np

from tests import TestEnv

from pyreinforce.brain import Brain
from pyreinforce.acting import EpsGreedyPolicy
from pyreinforce.memory import Memory
from pyreinforce import TdAgent


class TestBrain(Brain):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        seed = 123
        self._np_random = np.random.RandomState()
        self._np_random.seed(seed)

        self._weights = self._np_random.uniform(size=(n_inputs, n_outputs))

    def predict_q(self, states, **kwargs):
        if states.ndim < 2:
            states = np.expand_dims(states, axis=0)

        return np.matmul(states, self._weights)

    def train(self, states, actions, targets, **kwargs):
        batch_size = states.shape[0]

        expected_states_shape = (batch_size, self._n_inputs)
        expected_actions_shape = (batch_size, 1)
        expected_targets_shape = (batch_size, 1)

        assert states.shape == expected_states_shape,\
            'States shape. expected: {}, actual: {}'.format(expected_states_shape, states.shape)
        assert actions.shape == expected_actions_shape,\
            'Actions shape. expected: {}, actual: {}'.format(expected_actions_shape, actions.shape)
        assert targets.shape == expected_targets_shape,\
            'Targets shape. expected: {}, actual: {}'.format(expected_targets_shape, targets.shape)

        global_step = kwargs['global_step']
        train_freq = kwargs['train_freq']

        assert global_step % train_freq == 0,\
            'Train frequency not honored. global_step: {}, train_freq: {}'.format(global_step, train_freq)


class TdAgentTest(unittest.TestCase):
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

    def test_seed_set(self):
        seed = 123

        env1 = TestEnv()
        env1.seed(seed)
        brain1 = TestBrain(self._n_states, self._n_actions)
        acting1 = EpsGreedyPolicy(self._eps)
        replay_memory1 = Memory(self._capacity, self._batch_size)
        agent1 = TdAgent(self._n_episodes, env1, brain1, acting1,
                         replay_memory1, self._gamma)
        agent1.seed(seed)

        rewards1, _ = agent1.run()

        env2 = TestEnv()
        env2.seed(seed)
        brain2 = TestBrain(self._n_states, self._n_actions)
        acting2 = EpsGreedyPolicy(self._eps)
        replay_memory2 = Memory(self._capacity, self._batch_size)
        agent2 = TdAgent(self._n_episodes, env2, brain2, acting2,
                         replay_memory2, self._gamma)
        agent2.seed(seed)

        rewards2, _ = agent2.run()

        self.assertListEqual(rewards1, rewards2)

    def test_seed_not_set(self):
        seed = 123

        env1 = TestEnv()
        env1.seed(seed)
        brain1 = TestBrain(self._n_states, self._n_actions)
        acting1 = EpsGreedyPolicy(self._eps)
        replay_memory1 = Memory(self._capacity, self._batch_size)
        agent1 = TdAgent(self._n_episodes, env1, brain1, acting1,
                         replay_memory1, self._gamma)

        rewards1, _ = agent1.run()

        env2 = TestEnv()
        env2.seed(seed)
        brain2 = TestBrain(self._n_states, self._n_actions)
        acting2 = EpsGreedyPolicy(self._eps)
        replay_memory2 = Memory(self._capacity, self._batch_size)
        agent2 = TdAgent(self._n_episodes, env2, brain2, acting2,
                         replay_memory2, self._gamma)

        rewards2, _ = agent2.run()

        self.assertFalse(np.array_equal(rewards1, rewards2))

    def test_run_without_validation(self):
        validation_freq = None
        validation_episodes = None
        agent = TdAgent(self._n_episodes, self._env, self._brain, self._acting,
                        self._replay_memory, self._gamma, self._train_freq,
                        validation_freq, validation_episodes)
        rewards, _ = agent.run()

        self.assertEqual(self._n_episodes, len(rewards))

    def test_run_with_validation(self):
        validation_freq = 2
        validation_episodes = 10
        agent = TdAgent(self._n_episodes, self._env, self._brain, self._acting,
                        self._replay_memory, self._gamma, self._train_freq,
                        validation_freq, validation_episodes)
        rewards, _ = agent.run()
        rewards = np.array(rewards)

        n_validations = self._n_episodes // validation_freq
        self.assertSequenceEqual([n_validations, validation_episodes], rewards.shape)

    def test_act_training(self):
        n_actions = 500
        brain = TestBrain(self._n_states, n_actions)
        eps = 0.1
        acting = EpsGreedyPolicy(eps)
        agent = TdAgent(self._n_episodes, self._env, brain, acting,
                        self._replay_memory, self._gamma)

        n_steps = 10000
        s = np.random.uniform(size=(n_steps, self._n_states))
        q = agent._predict_q(s)
        eps_greedy_actions = [agent._act(s[i], False) for i in range(n_steps)]
        eps_greedy_actions = np.array(eps_greedy_actions)
        greedy_actions = np.argmax(q, axis=1)
        n_random_actions =  np.count_nonzero(greedy_actions - eps_greedy_actions)
        fraction_of_random_actions = n_random_actions / n_steps

        self.assertAlmostEqual(eps, fraction_of_random_actions, delta=0.01)

    def test_act_validation(self):
        agent = TdAgent(self._n_episodes, self._env, self._brain, self._acting,
                        self._replay_memory, self._gamma, self._train_freq)

        n_steps = 10000
        s = np.random.uniform(size=(n_steps, 3))
        q = agent._predict_q(s)
        actual_actions = [agent._act(s[i], True) for i in range(n_steps)]
        expected_actions = np.argmax(q, axis=1).tolist()

        self.assertListEqual(expected_actions, actual_actions)


if __name__ == '__main__':
    unittest.main()
