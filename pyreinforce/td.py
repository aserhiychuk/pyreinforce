import numpy as np

from pyreinforce.memory import Memory
from pyreinforce.core import SimpleAgent


class TdAgent(SimpleAgent):
    '''
    TODO Temporal Difference Agent class
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
        q = self._predict_q(s)
        a = self._acting.act(q, i=kwargs['i'], n_episodes=self._n_episodes)

        return a

    def _predict_q(self, states):
        q = self._brain.predict_q(states)

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
        states = np.stack(batch[:, 0])
        states1 = np.stack(batch[:, 3])
        qs = self._predict_q(states)
        qs1 = self._predict_q(states1)
        targets = self._get_targets(batch, qs, qs1)

        self._brain.train(states, targets)

    def _get_targets(self, batch, q, q1):
        target = np.empty_like(q)
        target[:] = q
        a = np.int32(batch[:, 1])
        r = batch[:, 2]
        s1_mask = batch[:, 4]
        ind = np.arange(batch.shape[0])
        target[ind, a] = r + s1_mask * self._gamma * np.max(q1, axis=1)

        return target
