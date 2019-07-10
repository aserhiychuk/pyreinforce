import numpy as np

from pyreinforce.memory import Memory
from pyreinforce.core import SimpleAgent


class DdpgAgent(SimpleAgent):
    '''
    TODO DDPG Agent class
    '''
    def __init__(self, n_episodes, env, brain, acting, gamma, replay_memory_size, replay_batch_size, converter=None):
        super().__init__(n_episodes, env, converter)
        self._brain = brain
        self._acting = acting
        self._gamma = gamma
        self._replay_memory = Memory(replay_memory_size)
        self._replay_batch_size = replay_batch_size

    def seed(self, seed=None):
        super().seed(seed)

        self._acting.seed(seed)
        self._replay_memory.seed(seed)

    def _act(self, s, **kwargs):
        a = self._predict_a(s, False)
        a = self._acting.act(a, i=kwargs['i'])

        return a

    def _predict_a(self, states, is_target=False):
        a = self._brain.predict_a(states, is_target)

        assert not np.isnan(a).any(), 'A contains nan: {}'.format(a)
        assert not np.isinf(a).any(), 'A contains inf: {}'.format(a)

        return a

    def _predict_q(self, states, actions, is_target=False):
        q = self._brain.predict_q(states, actions, is_target)

        assert not np.isnan(q).any(), 'Q contains nan: {}'.format(q)
        assert not np.isinf(q).any(), 'Q contains inf: {}'.format(q)

        return q

    def _observe(self, experience):
        self._replay_memory.add(experience)

        batch = self._replay_memory.sample(self._replay_batch_size)

        if len(batch) > 0:
            self._train(batch)

    def _train(self, batch):
        batch = np.array(batch)
        s = np.stack(batch[:, 0])
        a = np.stack(batch[:, 1])
        r = np.vstack(batch[:, 2])
        s1 = np.stack(batch[:, 3])
        s1_mask = np.vstack(batch[:, 4])

        a1 = self._predict_a(s1, True)
        q1 = self._predict_q(s1, a1, True)

        # target
        t = r + s1_mask * self._gamma * q1

        self._brain.train(s, a, t)
