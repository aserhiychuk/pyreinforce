import numpy as np

from pyreinforce.memory import Memory
from pyreinforce.core import SimpleAgent
from pyreinforce.utils import discount_rewards


class MonteCarloAgent(SimpleAgent):
    '''
    TODO Monte Carlo Agent class
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

        batch = self._replay_memory.sample(self._replay_batch_size)

        if len(batch) > 0:
            self._train(batch)

    def _train(self, batch):
        batch = np.array(batch)
        states = np.stack(batch[:, 0])
        qs = self._predict_q(states)
        returns = self._get_targets(batch, qs)

        self._brain.train(states, returns)

    def _get_targets(self, batch, q):
        target = np.empty_like(q)
        target[:] = q
        a = np.int32(batch[:, 1])
        g = batch[:, 2]
        ind = np.arange(batch.shape[0])
        target[ind, a] = g

        return target

    def _after_episode(self):
        self._replay_memory.flush(self._preprocess_episode)

    def _preprocess_episode(self, episode):
        episode = np.array(episode)
        episode[:, 2] = discount_rewards(episode[:, 2], self._gamma)

        return episode.tolist()
