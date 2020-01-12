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
                Current episode number.
            n_episodes : int
                Total number of episodes.
            rewards : list
                List of cumulative rewards obtained during prior episodes.
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

            if len(batch) > 0:
                self._train(batch)

    def _train(self, batch):
        """Perform a training step.

        Parameters
        ----------
        batch : list
            List of tuples (`s`, `a`, `r`, `s1`, `terminal_flag`).
        """
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
        """Compute TD targets for current states `s`.

        Parameters
        ----------
        batch : list
            List of tuples (`s`, `a`, `r`, `s1`, `terminal_flag`).
        q : array
            `Q`-values for current states `s`.
        q1 : array
            `Q`-values for next states `s1`.
        """
        target = np.empty_like(q)
        target[:] = q
        a = np.int32(batch[:, 1])
        r = batch[:, 2]
        s1_mask = 1 - batch[:, 4]
        ind = np.arange(batch.shape[0])
        target[ind, a] = r + s1_mask * self._gamma * np.max(q1, axis=1)

        return target

    def _after_episode(self):
        """Make sure all experiences are stored in the replay memory."""
        self._replay_memory.flush()
