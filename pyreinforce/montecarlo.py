import numpy as np

from pyreinforce.core import SimpleAgent


class MonteCarloAgent(SimpleAgent):
    """Agent that implements Monte Carlo algorithm."""

    def __init__(self, n_episodes, env, brain, acting, replay_memory, gamma,
                 train_freq=1, validation_freq=None, validation_episodes=None,
                 converter=None, callback=None):
        """
        Parameters
        ----------
        n_episodes : int
            Number of episodes to train the agent for.
        env : obj
            Environment.
        brain : Brain
            DQN.
        acting : ActingPolicy
            Action selection policy.
        replay_memory : Memory
            Experience replay memory.
        gamma : float
            Discount factor, must be between 0 and 1.
        train_freq : int, optional
            Training frequency.
        validation_freq : int, optional
            Specifies how many episodes to run before a new validation run is performed.
        validation_episodes : int, optional
            Number of episodes in each validation run.
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
        super().__init__(n_episodes, env, validation_freq, validation_episodes,
                         converter, callback)
        self._brain = brain
        self._acting = acting
        self._replay_memory = replay_memory
        self._gamma = gamma
        self._train_freq = train_freq

    def seed(self, seed=None):
        """Seed action selection policy and experience replay memory.

        Parameters
        ----------
        seed : int, optional
            Seed for random number generator.
        """
        super().seed(seed)

        self._acting.seed(seed)
        self._replay_memory.seed(seed)

    def _act(self, s, validation=False, **kwargs):
        """Choose action given current state of the environment.

        Parameters
        ----------
        s : obj
            Current state of the environment.
        validation : bool, optional
            Indicator that is True during validation.
        **kwargs
            cur_step : int, optional
                Current step within episode.
            cur_episode : int, optional
                Current episode.
            n_episodes : int, optional
                Total number of episodes.
            global_step : int, optional
                Global step across all episodes.

        Returns
        -------
        int
            Action according to action selection policy.
        """
        q = self._predict_q(s, validation=validation, **kwargs)

        if validation:
            a = np.argmax(q)
        else:
            a = self._acting.act(q, **kwargs)

        return a

    def _predict_q(self, states, **kwargs):
        """Predict `Q`-values given states.

        Parameters
        ----------
        states : array
            Array of states.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        array
            Array of `Q`-values for each state.

        Raises
        ------
        AssertionError
            If `Q`-values contain nan or inf.
        """
        q = self._brain.predict_q(states, **kwargs)

        assert not np.isnan(q).any(), 'Q contains nan: {}'.format(q)
        assert not np.isinf(q).any(), 'Q contains inf: {}'.format(q)

        return q

    def _observe(self, experience):
        """Store experience in the episode buffer and perform a training step.

        Parameters
        ----------
        experience : tuple
            Tuple of (`s`, `a`, `r`, `s1`, `terminal_flag`).
        """
        self._replay_memory.add(experience, buffer=True)

        if self._global_step % self._train_freq == 0:
            batch = self._replay_memory.sample()

            if batch is not None:
                self._train(batch)

    def _train(self, batch):
        """Perform a training step.

        Parameters
        ----------
        batch : tuple of arrays
            Tuple of `states`, `actions`, `rewards`, `next states`, `next states masks`.
        """
        s, a, g, _, _ = batch

        self._brain.train(s, a, g, global_step=self._global_step,
                          train_freq=self._train_freq)

    def _after_episode(self, episode_no, reward):
        """Compute discounted rewards and flush the episode buffer.

        Parameters
        ----------
        episode_no : int
            Episode number.
        reward : float
            Episode reward.
        """
        self._replay_memory.flush(self._gamma)

        super()._after_episode(episode_no, reward)
