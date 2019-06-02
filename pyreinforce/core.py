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
    def __init__(self, n_episodes, env, preprocess_state=None):
        super().__init__()
        self._n_episodes = n_episodes
        self._env = env
        self._preprocess_state = preprocess_state

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

        rewards = np.array(rewards, np.float32)
        stats = np.array(stats)
        stats = stats.min(), stats.max(), stats.mean(), stats.std()

        return rewards, stats

    def _run_episode(self, i):
        reward = 0
        done = False
        s = self._env.reset()

        if self._preprocess_state:
            s = self._preprocess_state(s) 

        while not done:
            a = self._act(s, i=i)
            s1, r, done, _ = self._env.step(a)

            if self._preprocess_state:
                s1 = self._preprocess_state(s1) 

            reward += r

            experience = (s, a, r, s1, 0 if done else 1)
            self._observe(experience)

            s = s1

        return reward

    def _before_episode(self, i=0):
        pass

    def _act(self, s, **kwargs):
        pass

    def _observe(self, experience):
        pass

    def _after_episode(self):
        pass