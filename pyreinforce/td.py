import numpy as np

from pyreinforce.core import SimpleAgent


class TdAgent(SimpleAgent):
    '''
    TODO Temporal Difference Agent class
    '''
    def __init__(self, n_episodes, env, brain, acting, replay_memory, gamma,
                 converter=None, train_freq=1, callback=None):
        super().__init__(n_episodes, env, converter, callback)
        self._brain = brain
        self._acting = acting
        self._replay_memory = replay_memory
        self._gamma = gamma
        self._train_freq = train_freq

    def seed(self, seed=None):
        super().seed(seed)

        self._acting.seed(seed)
        self._replay_memory.seed(seed)

    def _act(self, s, cur_step=0, cur_episode=0):
        q = self._predict_q(s, cur_step=cur_step)
        a = self._acting.act(q, cur_step=cur_step, cur_episode=cur_episode, n_episodes=self._n_episodes)

        return a

    def _predict_q(self, states, **kwargs):
        q = self._brain.predict_q(states, **kwargs)

        assert not np.isnan(q).any(), 'Q contains nan: {}'.format(q)
        assert not np.isinf(q).any(), 'Q contains inf: {}'.format(q)

        return q

    def _observe(self, experience):
        self._replay_memory.add(experience)

        batch = self._replay_memory.sample()

        if len(batch) > 0 and self._global_step % self._train_freq == 0:
            self._train(batch)

    def _train(self, batch):
        batch = np.array(batch)
        batch = np.reshape(batch, (-1, batch.shape[-1]))

        states = np.stack(batch[:, 0])
        states1 = np.stack(batch[:, 3])
        qs = self._predict_q(states)
        qs1 = self._predict_q(states1)
        targets = self._get_targets(batch, qs, qs1)

        self._brain.train(states, targets, global_step=self._global_step,
                          train_freq=self._train_freq)

    def _get_targets(self, batch, q, q1):
        target = np.empty_like(q)
        target[:] = q
        a = np.int32(batch[:, 1])
        r = batch[:, 2]
        s1_mask = batch[:, 4]
        ind = np.arange(batch.shape[0])
        target[ind, a] = r + s1_mask * self._gamma * np.max(q1, axis=1)

        return target

    def _after_episode(self):
        self._replay_memory.flush()
