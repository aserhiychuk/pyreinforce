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
