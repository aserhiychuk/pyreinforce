import numpy as np

from pyreinforce.distributed import DistributedAgent, WorkerAgent


class AsyncTdAgent(DistributedAgent):
    def __init__(self, n_episodes, env, brain, acting, gamma, train_freq,
                 validation_freq=None, validation_episodes=None,
                 converter=None, callback=None, n_workers=None):
        super().__init__(n_episodes, env, brain, train_freq, validation_freq,
                         validation_episodes, converter, callback, n_workers)

        self._acting = acting
        self._gamma = gamma

    def _create_worker(self, worker_no, conn_to_parent, shared_weights, barrier,
                       env, brain, *args, **kwargs):
        _acting = self._acting(worker_no) if callable(self._acting) else self._acting
        validation_episodes = self._validation_episodes[worker_no] if self._validation_freq else None

        worker = TdWorker(worker_no, conn_to_parent, shared_weights, barrier,
                          self._n_episodes, env, brain, _acting, self._gamma, self._train_freq,
                          self._validation_freq, validation_episodes, self._converter,
                          self._callback)

        return worker


class TdWorker(WorkerAgent):
    def __init__(self, worker_no, conn_to_parent, shared_weights, barrier,
                 n_episodes, env, brain, acting, gamma, train_freq,
                 validation_freq=None, validation_episodes=None, converter=None, callback=None):

        super().__init__(worker_no, conn_to_parent, shared_weights, barrier,
                         n_episodes, env, brain, train_freq, validation_freq, validation_episodes,
                         converter, callback)

        self._acting = acting
        self._gamma = gamma

    def _act(self, s, validation=False, **kwargs):
        q = self._predict_q(s, validation=validation, **kwargs)

        if validation:
            a = np.argmax(q)
        else:
            a = self._acting.act(q, **kwargs)

        return a

    def _predict_q(self, states, **kwargs):
        q = self._brain.predict_q(states, **kwargs)

        assert not np.isnan(q).any(), f'Q contains nan: {q}'
        assert not np.isinf(q).any(), f'Q contains inf: {q}'

        return q

    def _compute_grads(self, batch):
        s, a, r, s1, s1_mask = batch
        t = self._get_td_targets(r, s1, s1_mask)

        grads = self._brain.compute_gradients(s, a, t)

        return grads

    def _get_td_targets(self, r, s1, s1_mask):
        q1 = self._predict_q(s1)

        t = r + s1_mask * self._gamma * np.max(q1, axis=1, keepdims=True)

        return t
