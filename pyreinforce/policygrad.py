import numpy as np

from pyreinforce.core import SimpleAgent
from pyreinforce.utils import discount_rewards


class PolicyGradientAgent(SimpleAgent):
    '''
    TODO Policy Gradient Agent class
    '''
    def __init__(self, n_episodes, env, brain, acting, gamma, converter=None):
        super().__init__(n_episodes, env, converter)
        self._brain = brain
        self._acting = acting
        self._gamma = gamma
        self._episode_memory = []

    def seed(self, seed=None):
        super().seed(seed)

        self._acting.seed(seed)

    def _act(self, s, cur_step=0, cur_episode=0):
        probs = self._predict_policy(s, cur_step=cur_step)
        a = self._acting.act(probs)

        return a

    def _predict_policy(self, states, **kwargs):
        probs = self._brain.predict_policy(states, **kwargs)

        assert not np.isnan(probs).any(), 'policy contains nan: {}'.format(probs)
        assert not np.isinf(probs).any(), 'policy contains inf: {}'.format(probs)

        return probs

    def _observe(self, experience):
        self._episode_memory.append(experience)

    def _train(self, batch):
        states = np.stack(batch[:, 0])
        actions = batch[:, 1].reshape(-1, 1)
        returns = batch[:, 2].reshape(-1, 1)

        self._brain.train(states, actions, returns)

    def _after_episode(self):
        episode = np.array(self._episode_memory)
        episode[:, 2] = discount_rewards(episode[:, 2], self._gamma)

        self._train(episode)

        self._episode_memory = []
