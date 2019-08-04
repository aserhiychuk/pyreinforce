import numpy as np

from pyreinforce.core import SimpleAgent


class DdpgAgent(SimpleAgent):
    '''
    TODO DDPG Agent class
    '''
    def __init__(self, n_episodes, env, brain, acting, replay_memory, gamma, converter=None):
        super().__init__(n_episodes, env, converter)
        self._brain = brain
        self._acting = acting
        self._replay_memory = replay_memory
        self._gamma = gamma

    def seed(self, seed=None):
        super().seed(seed)

        self._acting.seed(seed)
        self._replay_memory.seed(seed)

    def _act(self, s, cur_step=0, cur_episode=0):
        a = self._predict_a(s, False, cur_step=cur_step)
        a = self._acting.act(a, cur_step=cur_step)

        return a

    def _predict_a(self, states, is_target=False, **kwargs):
        a = self._brain.predict_a(states, is_target, **kwargs)

        assert not np.isnan(a).any(), 'A contains nan: {}'.format(a)
        assert not np.isinf(a).any(), 'A contains inf: {}'.format(a)

        return a

    def _predict_q(self, states, actions, is_target=False, **kwargs):
        q = self._brain.predict_q(states, actions, is_target, **kwargs)

        assert not np.isnan(q).any(), 'Q contains nan: {}'.format(q)
        assert not np.isinf(q).any(), 'Q contains inf: {}'.format(q)

        return q

    def _observe(self, experience):
        self._replay_memory.add(experience)

        batch = self._replay_memory.sample()

        if len(batch) > 0:
            self._train(batch)

    def _train(self, batch):
        batch = np.array(batch)
        batch = np.reshape(batch, (-1, batch.shape[-1]))

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

    def _after_episode(self):
        self._replay_memory.flush()
