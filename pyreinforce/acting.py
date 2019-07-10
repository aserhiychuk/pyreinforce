import numpy as np


class ActingPolicy(object):
    '''
    TODO Acting Policy class
    '''
    def __init__(self):
        super().__init__()

        self.seed()

    def seed(self, seed=None):
        self._np_random = np.random.RandomState()
        self._np_random.seed(seed)

    def act(self, **kwargs):
        raise NotImplementedError()


class EpsGreedyPolicy(ActingPolicy):
    '''
    TODO Epsilon Greedy Policy class
    '''
    def __init__(self, eps):
        super().__init__()
        self._eps = eps

    def act(self, q, **kwargs):
        if self._np_random.uniform() < self._eps:
            n_actions = q.shape[1]
            a = self._np_random.choice(n_actions)
        else:
            a = np.argmax(q)

        return a


class DecayingEpsGreedyPolicy(EpsGreedyPolicy):
    '''
    TODO Decaying Epsilon Greedy Policy class
    '''
    def __init__(self, start_eps, end_eps, eps_decay):
        super().__init__(start_eps)
        self._start_eps = start_eps
        self._end_eps = end_eps
        self._eps_decay = eps_decay

    def act(self, q, **kwargs):
        i = kwargs['i']
        n_episodes = kwargs['n_episodes']
        self._eps = self._end_eps + (self._start_eps - self._end_eps) * (1 - i / n_episodes) ** self._eps_decay
        # self._eps = self._end_eps + (self._start_eps - self._end_eps) * np.exp(-self._eps_decay * i)

        return super().act(q, **kwargs)


class SoftmaxPolicy(ActingPolicy):
    '''
    TODO Softmax Policy class
    '''
    def __init__(self):
        super().__init__()

    def act(self, probs, **kwargs):
        n_actions = probs.shape[1]
        a = self._np_random.choice(n_actions, p=probs[0])

        return a


class OrnsteinUhlenbeckPolicy(ActingPolicy):
    '''
    TODO Ornstein-Uhlenbeck Policy class
    '''
    def __init__(self, shape, mu, theta, sigma):
        super().__init__()

        self._shape = shape
        self._mu = mu * np.ones(shape)
        self._theta = theta
        self._sigma = sigma
        self._state = None
        self._cur_episode = None

    def _reset(self):
        self._state = np.copy(self._mu)

    def _noise(self, **kwargs):
        x = self._state
        dx = self._theta * (self._mu - x) + self._sigma * self._np_random.standard_normal(self._shape)
        self._state = x + dx

        return self._state

    def act(self, a, **kwargs):
        cur_episode = kwargs['i']

        if self._cur_episode != cur_episode:
            self._cur_episode = cur_episode
            self._reset()

        return a[0] + self._noise()
