import numpy as np


class TestEnv:
    def __init__(self):
        self._cur_episode = None
        self._cur_step = None
        self._global_step = 0
        self._cur_episode_max_steps = None
        self.rewards = {}

        self.seed()

    def seed(self, seed=None):
        self._np_random = np.random.RandomState()
        self._np_random.seed(seed)

    def reset(self):
        self._cur_episode_max_steps = self._np_random.randint(10, 50)

        self._cur_episode = 0 if self._cur_episode is None else self._cur_episode + 1
        self._cur_step = 0

        s = self._get_current_state()

        return s

    def step(self, a):
        self._cur_step += 1
        self._global_step += 1

        s1 = self._get_current_state()
        r = a / 100
        done = (self._cur_step == self._cur_episode_max_steps)
        info = None

        if self._cur_episode not in self.rewards:
            self.rewards[self._cur_episode] = 0

        self.rewards[self._cur_episode] += r

        return s1, r, done, info

    def _get_current_state(self):
        state = (self._cur_episode, self._cur_step, self._global_step)
        state = np.array(state)

        return state
