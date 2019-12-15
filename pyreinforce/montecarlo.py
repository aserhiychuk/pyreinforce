import numpy as np

from pyreinforce.core import SimpleAgent


class MonteCarloAgent(SimpleAgent):
    '''
    TODO Monte Carlo Agent class
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
        self._replay_memory.add(experience, buffer=True)

        if self._global_step % self._train_freq == 0:
            batch = self._replay_memory.sample()

            if len(batch) > 0:
                self._train(batch)

    def _train(self, batch):
        batch = np.array(batch)
        batch = np.reshape(batch, (-1, batch.shape[-1]))

        states = np.stack(batch[:, 0])
        qs = self._predict_q(states)
        returns = self._get_targets(batch, qs)

        self._brain.train(states, returns, global_step=self._global_step,
                          train_freq=self._train_freq)

    def _get_targets(self, batch, q):
        target = np.empty_like(q)
        target[:] = q
        a = np.int32(batch[:, 1])
        g = batch[:, 2]
        ind = np.arange(batch.shape[0])
        target[ind, a] = g

        return target

    def _after_episode(self):
        self._replay_memory.flush(self._gamma)
