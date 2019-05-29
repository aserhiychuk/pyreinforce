import numpy as np

from pyreinforce.core import SimpleAgent


class PolicyGradientAgent(SimpleAgent):
    '''
    TODO Policy Gradient Agent class
    '''
    def __init__(self, n_episodes, env, brain, acting, gamma, preprocess_state=None):
        super().__init__(n_episodes, env, preprocess_state)
        self._brain = brain
        self._acting = acting
        self._gamma = gamma
        self._episode_memory = []

    def _act(self, s, **kwargs):
        probs = self._predict_policy(s)
        a = self._acting.act(probs)

        return a

    def _predict_policy(self, states):
        probs = self._brain.predict_policy(states)

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
        episode[:, 2] = self._discount_rewards(episode[:, 2])

        self._train(episode)

        self._episode_memory = []

    def _discount_rewards(self, rewards):
        result = np.empty_like(rewards, dtype=np.float32)
        g = 0

        for i in reversed(range(len(rewards))):
            g = rewards[i] + self._gamma * g
            result[i] = g

        return result
