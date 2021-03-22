import unittest
import random

import numpy as np

from tests import TestEnv as AbstractTestEnv

from pyreinforce.converter import Converter
from pyreinforce.core import SimpleAgent, Callback


class TestEnv(AbstractTestEnv):
    def __init__(self, test_case):
        super().__init__()

        self._test_case = test_case

    def step(self, a):
        # covers convert_action
        self._test_case.assertEqual(0, a % 10, f'Action. expected: divisible by 10, actual: {a}')

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


class TestCallback(Callback):
    def __init__(self, test_case, n_episodes, env, validation_freq, validation_episodes):
        self._test_case = test_case
        self._n_episodes = n_episodes
        self._env = env
        self._validation_freq = validation_freq
        self._validation_episodes = validation_episodes

        self._is_running = False
        self._in_validation = False
        self._current_episode_no = -1
        self._current_global_step = 0
        self._n_validations = 0
        self._prev_s = None

    def on_before_run(self, **kwargs):
        self._is_running = True

        n_episodes = kwargs['n_episodes']
        self._test_case.assertEqual(self._n_episodes, n_episodes)

    def on_after_run(self, **kwargs):
        self._is_running = False

        if self._validation_freq is not None:
            expected_n_validations = self._n_episodes // self._validation_freq
            self._test_case.assertEqual(expected_n_validations, self._n_validations,
                                        f'# of validations. expected: {expected_n_validations}, actual: {self._n_validations}')

    def on_state_change(self, s, **kwargs):
        self._test_case.assertTrue(self._is_running)

        episode_no, step_no, global_step = s

        if self._in_validation and step_no == 0:
            # prev_s is not automatically reset between validation episodes
            self._prev_s = None

        if self._prev_s is None:
            self._test_case.assertEqual(0, step_no)
        else:
            prev_episode_no, prev_step_no, prev_global_step = self._prev_s
            self._test_case.assertEqual(prev_episode_no, episode_no)
            self._test_case.assertEqual(prev_step_no + 1, step_no)
            self._test_case.assertEqual(prev_global_step + 1, global_step)

        self._prev_s = s

    def on_before_episode(self, episode_no, **kwargs):
        self._test_case.assertTrue(self._is_running)

        self._current_episode_no += 1
        self._prev_s = None

    def on_after_episode(self, episode_no, reward, **kwargs):
        self._test_case.assertTrue(self._is_running)

        self._test_case.assertEqual(self._current_episode_no, episode_no)
        self._test_case.assertLessEqual(episode_no, self._n_episodes - 1)

        if self._validation_freq is None:
            self._test_case.assertEqual(episode_no, len(self._env.rewards) - 1)

            expected_reward = self._env.rewards[episode_no]
            self._test_case.assertEqual(expected_reward, reward)

        global_step = kwargs['global_step']
        self._test_case.assertLess(self._current_global_step, global_step)
        self._current_global_step = global_step

    def on_before_validation(self, **kwargs):
        self._in_validation = True

        if self._validation_freq is None:
            self._test_case.fail('Validation is disabled')

        self._n_validations += 1

    def on_after_validation(self, rewards, **kwargs):
        self._in_validation = False

        if self._validation_freq is None:
            self._test_case.fail('Validation is disabled')

        self._test_case.assertEqual(len(rewards), self._validation_episodes,
                                    f'# of validation episodes. expected <= {self._validation_episodes}, actual: {len(rewards)}')


class TestAgent(SimpleAgent):
    def __init__(self, test_case, n_episodes, env, validation_freq=None, validation_episodes=None,
                 converter=None, callback=None):
        super().__init__(n_episodes, env, validation_freq, validation_episodes,
                         converter, callback)

        self._test_case = test_case

    def _act(self, s, validation=False, **kwargs):
        if self._validation_freq is None:
            expected_cur_episode, expected_cur_step, expected_global_step, _ = s
            self._test_case.assertEqual(expected_cur_episode, kwargs['cur_episode'])
            self._test_case.assertEqual(expected_cur_step, kwargs['cur_step'])
            self._test_case.assertEqual(expected_global_step, self._global_step)

        return random.randint(-5, 5)

    def _observe(self, experience):
        super()._observe(experience)

        s, a, r, _, _ = experience

        # covers convert_state
        self._test_case.assertEqual(4, len(s), f'State length. expected: {4}, actual: {len(s)}')

        if self._validation_freq is None:
            _, _, expected_global_step, _ = s
            self._test_case.assertEqual(expected_global_step, self._global_step,\
                                        'Global step. expected: {expected_global_step}, actual: {self._global_step}')

        # experience contains original action
        self._test_case.assertLessEqual(-5, a, f'Action. expected: -5 <= a, actual: {a}')
        self._test_case.assertLessEqual(a, 5, f'Action. expected: a <= 5, actual: {a}')

        # covers convert_experience
        if a < 0:
            expected_reward = -1
        elif a > 0:
            expected_reward = 1
        else:
            expected_reward = 0

        self._test_case.assertEqual(expected_reward, r, f'Reward. expected: {expected_reward}, actual: {r}')


class SimpleAgentTest(unittest.TestCase):
    def setUp(self):
        self._n_episodes = 10
        self._env = TestEnv(self)
        self._converter = TestConverter()

    def test_run_without_validation(self):
        validation_freq =  None
        validation_episodes = None
        callback = TestCallback(self, self._n_episodes, self._env, validation_freq, validation_episodes)

        self._agent = TestAgent(self, self._n_episodes, self._env,
                                validation_freq, validation_episodes,
                                self._converter, callback)

        rewards, _ = self._agent.run()

        self.assertEqual(self._n_episodes, len(rewards))

        for episode, actual_reward in enumerate(rewards):
            expected_reward = self._env.rewards[episode]
            self.assertAlmostEqual(expected_reward, actual_reward, places=4)

    def test_run_with_validation(self):
        validation_freq =  2
        validation_episodes = 3
        callback = TestCallback(self, self._n_episodes, self._env, validation_freq, validation_episodes)

        self._agent = TestAgent(self, self._n_episodes, self._env,
                                validation_freq, validation_episodes,
                                self._converter, callback)

        rewards, _ = self._agent.run()
        rewards = np.array(rewards)

        n_validations = self._n_episodes // validation_freq
        self.assertSequenceEqual([n_validations, validation_episodes], rewards.shape)


if __name__ == '__main__':
    unittest.main()
