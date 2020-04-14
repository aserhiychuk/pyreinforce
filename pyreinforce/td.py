import numpy as np

from pyreinforce.core import SimpleAgent


class TdAgent(SimpleAgent):
    """Agent that implements Temporal Difference algorithm."""

    def __init__(self, n_episodes, env, brain, acting, replay_memory, gamma,
                 converter=None, train_freq=1, callback=None):
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
        converter : Converter, optional
            If specified, allows to pre/post process state, action, or experience.
        train_freq : int, optional
            Training frequency.
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
        q = self._predict_q(s, cur_step=cur_step)
        a = self._acting.act(q, cur_step=cur_step, cur_episode=cur_episode,
                             n_episodes=self._n_episodes, global_step=self._global_step)

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
        """Store experience in the replay memory and perform a training step.

        Parameters
        ----------
        experience : tuple
            Tuple of (`s`, `a`, `r`, `s1`, `terminal_flag`).
        """
        self._replay_memory.add(experience)

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
        s, a, r, s1, s1_mask = batch

        t = self._get_td_targets(r, s1, s1_mask)

        self._brain.train(s, a, t, global_step=self._global_step,
                          train_freq=self._train_freq)

    def _get_td_targets(self, r, s1, s1_mask):
        """Compute TD targets for current states `s`.

        Parameters
        ----------
        r : array
            Rewards.
        s1 : array
            Next states.
        s1_mask : array
            Next states masks.

        Returns
        -------
        array
            TD targets.
        """
        q1 = self._predict_q(s1)

        target = r + s1_mask * self._gamma * np.max(q1, axis=1)

        return target

    def _after_episode(self):
        """Make sure all experiences are stored in the replay memory."""
        self._replay_memory.flush()
