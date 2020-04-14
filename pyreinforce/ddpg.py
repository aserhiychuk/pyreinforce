import numpy as np

from pyreinforce.core import SimpleAgent


class DdpgAgent(SimpleAgent):
    """Agent that implements Deep Deterministic Policy Gradient algorithm."""

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
            Holder for both actor and critic.
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
        a = self._predict_a(s, False, cur_step=cur_step)
        a = self._acting.act(a, cur_step=cur_step)

        return a

    def _predict_a(self, states, is_target=False, **kwargs):
        """Predict actions given states.

        Parameters
        ----------
        states : array
            Array of states.
        is_target : bool, optional
            Use primary network if False, and target one otherwise.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        array
            Array of actions.

        Raises
        ------
        AssertionError
            If actions contain nan or inf.
        """
        a = self._brain.predict_a(states, is_target, **kwargs)

        assert not np.isnan(a).any(), 'A contains nan: {}'.format(a)
        assert not np.isinf(a).any(), 'A contains inf: {}'.format(a)

        return a

    def _predict_q(self, states, actions, is_target=False, **kwargs):
        """Predict `Q`-values given states and actions.

        Parameters
        ----------
        states : array
            Array of states.
        actions : array
            Array of actions.
        is_target : bool, optional
            Use primary network if False, and target one otherwise.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        array
            Array of `Q`-values.

        Raises
        ------
        AssertionError
            If `Q`-values contain nan or inf.
        """
        q = self._brain.predict_q(states, actions, is_target, **kwargs)

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

        a = np.reshape(a, (-1, 1))
        r = np.reshape(r, (-1, 1))
        s1_mask = np.reshape(s1_mask, (-1, 1))

        a1 = self._predict_a(s1, True)
        q1 = self._predict_q(s1, a1, True)

        # target
        t = r + s1_mask * self._gamma * q1

        self._brain.train(s, a, t, global_step=self._global_step,
                          train_freq=self._train_freq)

    def _after_episode(self):
        """Make sure all experiences are stored in the replay memory."""
        self._replay_memory.flush()
