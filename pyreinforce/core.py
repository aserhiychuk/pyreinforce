import time

import numpy as np


class Agent(object):
    '''
    TODO Agent class
    '''
    def __init__(self):
        super().__init__()
#         self._logger = logging.getLogger('{}.{}'.format(__name__, type(self).__name__))

    def seed(self, seed=None):
        pass

    def run(self):
        pass


class SimpleAgent(Agent):
    '''
    TODO Simple Agent class
    '''
    def __init__(self, n_episodes, env, converter=None, callback=None):
        super().__init__()
        self._n_episodes = n_episodes
        self._env = env
        self._converter = converter
        self._callback = callback

        self._global_step = 0

    def run(self):
        rewards = []
        stats = []

        for i in range(self._n_episodes):
            episode_start = time.perf_counter()

            self._before_episode(i)
            reward = self._run_episode(i)
            self._after_episode()

            episode_stop = time.perf_counter()
            stats.append(episode_stop - episode_start)

            rewards.append(reward)

            if callable(self._callback):
                self._callback(i, self._n_episodes, rewards)

        rewards = np.array(rewards, np.float32)
        stats = np.array(stats)
        stats = stats.min(), stats.max(), stats.mean(), stats.std()

        return rewards, stats

    def _run_episode(self, i):
        cur_step = 0
        reward = 0
        done = False
        s = self._reset()

        if self._converter:
            s = self._converter.convert_state(s)

        while not done:
            a = self._act(s, cur_step, i)

            if self._converter:
                a = self._converter.convert_action(a)

            s1, r, done, _ = self._step(a)

            if self._converter:
                s1 = self._converter.convert_state(s1)
                r = self._converter.convert_reward(r)

            reward += r

            experience = (s, a, r, s1, 0 if done else 1)
            self._observe(experience)

            s = s1

            cur_step += 1
            self._global_step += 1

        return reward

    def _reset(self):
        s = self._env.reset()

        return s

    def _step(self, a):
        s1, r, done, info = self._env.step(a)

        return s1, r, done, info

    def _before_episode(self, i=0):
        pass

    def _act(self, s, cur_step=0, cur_episode=0):
        pass

    def _observe(self, experience):
        pass

    def _after_episode(self):
        pass
