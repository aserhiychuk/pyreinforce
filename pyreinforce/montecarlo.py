import numpy as np

from pyreinforce.memory import Memory
from pyreinforce.core import SimpleAgent


class MonteCarloAgent(SimpleAgent):
    '''
    TODO Monte Carlo Agent class
    '''
    def __init__(self, n_episodes, env, brain, acting, gamma, replay_memory_size, replay_batch_size):
        super().__init__(n_episodes, env)
        self._brain = brain
        self._acting = acting
        self._gamma = gamma
        self._episode_memory = []
        self._replay_memory = Memory(replay_memory_size)
        self._replay_batch_size = replay_batch_size

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
        self._episode_memory.append(experience)

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
        episode = np.array(self._episode_memory)
        episode[:, 2] = self._discount_rewards(episode[:, 2])
        episode = episode.tolist()

        self._replay_memory.add(episode)
        self._episode_memory = []

    def _discount_rewards(self, rewards):
        result = np.empty_like(rewards, dtype=np.float32)
        g = 0

        for i in reversed(range(len(rewards))):
            g = rewards[i] + self._gamma * g
            result[i] = g

        return result
