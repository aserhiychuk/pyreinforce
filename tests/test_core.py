import unittest
import random

import numpy as np

from tests import TestEnv as AbstractTestEnv

from pyreinforce.converter import Converter
from pyreinforce.core import SimpleAgent


class TestEnv(AbstractTestEnv):
    def step(self, a):
        # covers convert_action
        assert a % 10 == 0, 'Action. expected: divisible by 10, actual: {}'.format(a)

        return super().step(a)


class TestConverter(Converter):
    def convert_state(self, s, info=None):
        s = s.tolist()
        s = s + [None]
        s = np.array(s)

        return s

    def convert_action(self, a):
        return a * 10

    def convert_experience(self, experience, info=None):
        s, a, r, s1, is_terminal = experience

        if a < 0:
            r = -1
        elif a > 0:
            r = 1
        else:
            r = 0

        return s, a, r, s1, is_terminal


class TestAgent(SimpleAgent):
    def __init__(self, n_episodes, env, converter=None, callback=None):
        super().__init__(n_episodes, env, converter, callback)

    def _act(self, s, cur_step=0, cur_episode=0):
        expected_cur_episode, expected_cur_step, expected_global_step, _ = s
        assert expected_cur_episode == cur_episode
        assert expected_cur_step == cur_step
        assert expected_global_step == self._global_step

        return random.randint(-5, 5)

    def _observe(self, experience):
        super()._observe(experience)

        s, a, r, _, _ = experience

        # covers convert_state
        assert len(s) == 4, 'State length. expected: {}, actual: {}'.format(4, len(s))

        _, _, expected_global_step, _ = s
        assert expected_global_step == self._global_step, 'Global step. expected: {}, actual: {}'.format(expected_global_step,
                                                                                                         self._global_step)

        # experience contains original action
        assert -5 <= a and a <= 5, 'Action. expected: -5 <= a <= 5, actual: {}'.format(a)

        # covers convert_experience
        if a < 0:
            expected_reward = -1
        elif a > 0:
            expected_reward = 1
        else:
            expected_reward = 0

        assert r == expected_reward, 'Reward. expected: {}, actual: {}'.format(expected_reward, r)


class SimpleAgentTest(unittest.TestCase):
    def setUp(self):
        self._n_episodes = 10
        self._env = TestEnv()
        converter = TestConverter()
        episode_callback = self._create_episode_callback()
        self._agent = TestAgent(self._n_episodes, self._env,
                                converter, episode_callback)

    def _create_episode_callback(self):
        def callback(cur_episode, reward, **kwargs):
            rewards = kwargs['rewards']
            n_episodes = kwargs['n_episodes']
            n_episode_steps = kwargs['n_episode_steps']
            global_step = kwargs['global_step']

            self.assertEqual(reward, rewards[-1])
            self.assertEqual(self._n_episodes, n_episodes)
            self.assertLessEqual(n_episode_steps, global_step)

            self.assertLessEqual(cur_episode, self._n_episodes - 1)
            self.assertEqual(cur_episode, len(rewards) - 1)
            self.assertEqual(cur_episode, len(self._env.rewards) - 1)

            for episode, actual_reward in enumerate(rewards):
                expected_reward = self._env.rewards[episode]
                self.assertEqual(expected_reward, actual_reward)

        return callback

    def test_run(self):
        rewards, _ = self._agent.run()

        self.assertEqual(self._n_episodes, len(rewards))

        for episode, actual_reward in enumerate(rewards):
            expected_reward = self._env.rewards[episode]
            self.assertAlmostEqual(expected_reward, actual_reward, places=4)


if __name__ == '__main__':
    unittest.main()
