import numpy as np


class ActingPolicy(object):
    """Base class for all action selection policies."""

    def __init__(self):
        super().__init__()

        self.seed()

    def seed(self, seed=None):
        """Seed the random number generator.

        Parameters
        ----------
        seed : int, optional
            Seed for random number generator.
        """
        self._np_random = np.random.RandomState()
        self._np_random.seed(seed)

    def act(self, **kwargs):
        """
        Select action given current state of the environment.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments.

        Note
        ----
        Subclasses must implement this method.
        """
        raise NotImplementedError


class EpsGreedyPolicy(ActingPolicy):
    """Select action according to epsilon greedy policy."""

    def __init__(self, eps):
        """
        Parameters
        ----------
        eps : float
            Probability of a random action.
        """
        super().__init__()
        self._eps = eps

    def act(self, q, **kwargs):
        """Select action based on `Q`-values.

        Parameters
        ----------
        q : array_like
            `1 x N` array of `Q`-values.

        Returns
        -------
        int
            Random action with probability `self._eps`, index of
            the maximum `Q`-value otherwise.
        """
        if self._np_random.uniform() < self._eps:
            n_actions = q.shape[1]
            a = self._np_random.choice(n_actions)
        else:
            a = np.argmax(q)

        return a


class DecayingEpsGreedyPolicy(EpsGreedyPolicy):
    """Select action according to epsilon greedy policy with decaying epsilon."""

    def __init__(self, eps_start, eps_end, eps_decay):
        """
        Parameters
        ----------
        eps_start : float
            Initial probability of a random action.
        eps_end : float
            Final probability of a random action.
        eps_decay : float
            Rate with which probability of a random action
            is decayed from `eps_start` to `eps_end`.
        """
        super().__init__(eps_start)
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay = eps_decay

    def act(self, q, **kwargs):
        """Select action based on `Q`-values.

        Parameters
        ----------
        q : array_like
            `1 x N` array of `Q`-values.
        **kwargs
            cur_step : int
                Current step within episode.
            cur_episode: int
                Current episode.
            n_episodes : int
                Total number of episodes.
            global_step : int
                Global step across all episodes.

        Returns
        -------
        int
            Random action with probability `self._eps`, index of
            the maximum `Q`-value otherwise.
        """
        cur_step = kwargs['cur_step']

        if cur_step == 0:
            cur_episode = kwargs['cur_episode']
            n_episodes = kwargs['n_episodes']
            self._eps = self._eps_end + (self._eps_start - self._eps_end) * (1 - cur_episode / n_episodes) ** self._eps_decay
#             self._eps = self._eps_end + (self._eps_start - self._eps_end) * np.exp(-self._eps_decay * cur_episode)

        return super().act(q, **kwargs)


class CustomEpsGreedyPolicy(EpsGreedyPolicy):
    """Select action according to epsilon greedy policy with custom epsilon."""

    def __init__(self, get_eps):
        """
        Parameters
        ----------
        get_eps : func
            Calculates epsilon given where the agent is in the learning process.

            Keyword arguments
            -----------------
            cur_step : int
                Current step within episode.
            cur_episode: int
                Current episode.
            n_episodes : int
                Total number of episodes.
            global_step : int
                Global step across all episodes.
        """
        super().__init__(None)

        assert callable(get_eps), 'Expected function but got "{}"'.format(get_eps)

        self._get_eps = get_eps

    def act(self, q, **kwargs):
        """Select action based on `Q`-values.

        Parameters
        ----------
        q : array_like
            `1 x N` array of `Q`-values.
        **kwargs
            cur_step : int
                Current step within episode.
            cur_episode: int
                Current episode.
            n_episodes : int
                Total number of episodes.
            global_step : int
                Global step across all episodes.

        Returns
        -------
        int
            Random action with probability `self._eps`, index of
            the maximum `Q`-value otherwise.
        """
        self._eps = self._get_eps(**kwargs)

        return super().act(q, **kwargs)


class SoftmaxPolicy(ActingPolicy):
    """Softmax implementation of action selection policy."""

    def __init__(self):
        super().__init__()

    def act(self, probs, **kwargs):
        """
        Parameters
        ----------
        probs : array
            `1 x N` array of probabilities for each action.

        Returns
        -------
        int
            Random integer in range `0` to `N - 1`
            according to probabilities `probs`.
        """
        n_actions = probs.shape[1]
        a = self._np_random.choice(n_actions, p=probs[0])

        return a


class OrnsteinUhlenbeckPolicy(ActingPolicy):
    """Ornstein-Uhlenbeck process implementation of action selection policy.

    This policy implements Ornstein-Uhlenbeck process which modifies
    actions by adding time-correlated noise to them.
    """

    def __init__(self, shape, mu, theta, sigma):
        """
        Parameters
        ----------
        shape : int or tuple of ints
            Noise shape.
        mu : float
            The equilibrium or mean value.
        theta : float
            Determines how “fast” the variable reverts towards the mean.
        sigma : float
            Degree of volatility of the process.
        """
        super().__init__()
        self._shape = shape
        self._mu = mu * np.ones(shape)
        self._theta = theta
        self._sigma = sigma
        self._state = None

    def _reset(self):
        """Reset current state."""
        self._state = np.copy(self._mu)

    def _noise(self, **kwargs):
        """Calculate noise.

        Returns
        -------
        array
            Noise of shape `self._shape`
        """
        x = self._state
        dx = self._theta * (self._mu - x) + self._sigma * self._np_random.standard_normal(self._shape)
        self._state = x + dx

        return self._state

    def act(self, a, **kwargs):
        """
        Parameters
        ----------
        a : array
            `1 x N` array of actions.
        **kwargs
            cur_step : int
                Current step within episode.

        Returns
        -------
        array
            Action(s) with added noise.
        """
        cur_step = kwargs['cur_step']

        if cur_step == 0:
            self._reset()

        return a[0] + self._noise()
