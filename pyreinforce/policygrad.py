import numpy as np

from pyreinforce.core import SimpleAgent
from pyreinforce.utils import discount_rewards


class PolicyGradientAgent(SimpleAgent):
    """Agent that implements Policy Gradient algorithm."""

    def __init__(self, n_episodes, env, brain, acting, gamma,
                 converter=None, callback=None):
        """
        Parameters
        ----------
        n_episodes : int
            Number of episodes to train the agent for.
        env : obj
            Environment.
        brain : Brain
            Neural network.
        acting : ActingPolicy
            Action selection policy.
        gamma : float
            Discount factor, must be between 0 and 1.
        converter : Converter, optional
            If specified, allows to pre/post process state, action, or experience.
        callback : callable, optional
            If specified, is called after each episode
            with the following parameters:

            cur_episode : int
                Episode number.
            reward : float
                Episode cumulative reward.
            **kwargs
                Additional keyword arguments.
        """
        super().__init__(n_episodes, env, converter, callback)
        self._brain = brain
        self._acting = acting
        self._gamma = gamma
        self._episode_memory = []

    def seed(self, seed=None):
        """Seed action selection policy.

        Parameters
        ----------
        seed : int, optional
            Seed for random number generator.
        """
        super().seed(seed)

        self._acting.seed(seed)

    def _act(self, s, cur_step=0, cur_episode=0):
        """Choose action given current state of the environment.

        Parameters
        ----------
        s : obj
            Current state of the environment.
        cur_step : int, optional
            Current step within episode.
        cur_episode : int, optional
            Current episode.

        Returns
        -------
        int
            Action according to action selection policy.
        """
        probs = self._predict_policy(s, cur_step=cur_step)
        a = self._acting.act(probs)

        return a

    def _predict_policy(self, states, **kwargs):
        """Predict actions probabilities given states.

        Parameters
        ----------
        states : array
            Array of states.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        array
            Array of actions probabilities for each state.

        Raises
        ------
        AssertionError
            If actions probabilities contain nan or inf.
        """
        probs = self._brain.predict_policy(states, **kwargs)

        assert not np.isnan(probs).any(), 'policy contains nan: {}'.format(probs)
        assert not np.isinf(probs).any(), 'policy contains inf: {}'.format(probs)

        return probs

    def _observe(self, experience):
        """Store experience in the episode memory.

        Parameters
        ----------
        experience : tuple
            Tuple of (`s`, `a`, `r`, `s1`, `terminal_flag`).
        """
        self._episode_memory.append(experience)

    def _train(self, batch):
        """Perform a training step.

        Parameters
        ----------
        batch : array
            Array of experiences (`s`, `a`, `r`, `s1`, `terminal_flag`).
        """
        states = np.stack(batch[:, 0])
        actions = batch[:, 1].reshape(-1, 1)
        returns = batch[:, 2].reshape(-1, 1)

        self._brain.train(states, actions, returns)

    def _after_episode(self):
        """Compute discounted rewards and perform a training step."""
        episode = np.array(self._episode_memory)
        episode[:, 2] = discount_rewards(episode[:, 2], self._gamma)

        self._train(episode)

        self._episode_memory = []
