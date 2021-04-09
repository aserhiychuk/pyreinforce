import numpy as np

from tests import LinearQBrain


class DummyBrain:
    def __init__(self):
        self._expected_shapes = [(3, 4), (7, 5)]
        self._weights = [np.zeros(shape, np.float32) for shape in self._expected_shapes]

    def apply_gradients(self, grads):
        pass
        # TODO uncomment after implementing timeout in workers
#         actual_shapes = [g.shape for g in grads]
#
#         for e, a in zip(self._expected_shapes, actual_shapes):
#             assert e == a, f'Grads shape. expected: {e}, actual: {a}'

    def get_weights(self):
        return self._weights

    def set_weights(self, weights, context=None, **kwargs):
        pass


class DummyQBrain(DummyBrain):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()

        self._n_inputs = n_inputs
        self._n_outputs = n_outputs

    def predict_q(self, s, **kwargs):
        if s.ndim == 1:
            assert 'validation' in kwargs
            assert 'cur_step' in kwargs
            assert 'cur_episode' in kwargs
            assert 'n_episodes' in kwargs
            assert 'global_step' in kwargs

            s = np.expand_dims(s, axis=0)
        elif s.ndim == 2:
            assert not kwargs.keys()
        else:
            raise AssertionError(f'States shape. expected: 1 or 2 dimensions, actual: {s.shape}')

        batch_size = s.shape[0]

        return np.random.uniform(size=(batch_size, self._n_outputs))

    def compute_gradients(self, s, a, t, **kwargs):
        batch_size = s.shape[0]
        assert s.shape == (batch_size, self._n_inputs)
        assert a.shape == (batch_size, 1)
        assert t.shape == (batch_size, 1)

        return []


class DummyConn:
    def __init__(self, worker_no):
        self._worker_no = worker_no
        self._ref_id = None

    def send(self, msg):
        self._ref_id = msg['ref_id']
        assert msg['type'] in ['gradients', 'result']
        assert msg['worker_no'] == self._worker_no

    def recv(self):
        return {
            'type'  : 'weights',
            'ref_id': self._ref_id
        }


class DummySharedWeights:
    def __init__(self, metadata):
        self._metadata = metadata
        self._weights = [np.zeros(shape, dtype) for shape, dtype, _ in self._metadata]

    def read(self, worker_no):
        return self._weights

    def write(self, worker_no, weights):
        self._weights = weights


class AsyncLinearQBrain(LinearQBrain):
    def predict_q(self, s, **kwargs):
        if s.ndim == 1:
            assert 'validation' in kwargs
            assert 'cur_step' in kwargs
            assert 'cur_episode' in kwargs
            assert 'n_episodes' in kwargs
            assert 'global_step' in kwargs

            s = np.expand_dims(s, axis=0)
        elif s.ndim == 2:
            assert not kwargs.keys()
        else:
            raise AssertionError(f'States shape. 1 or 2 dimensions, actual: {s.shape}')

        return super().predict_q(s, **kwargs)

    def compute_gradients(self, s, a, t, **kwargs):
        batch_size = s.shape[0]
        # state is a pair of coordinates (i, j)
        assert s.shape == (batch_size, 2)
        assert a.shape == (batch_size, 1)
        assert t.shape == (batch_size, 1)

        grads = super().compute_gradients(s, a, t)

        for grad in grads:
            grad *= s.shape[0]

        return grads

    def apply_gradients(self, grads):
        # TODO uncomment after implementing timeout in workers
#         actual_shapes = [g.shape for g in grads]
#
#         for e, a in zip(self._expected_shapes, actual_shapes):
#             assert e == a, f'Grads shape. expected: {e}, actual: {a}'

        super().apply_gradients(grads)

    def set_weights(self, weights, context=None, **kwargs):
        super().set_weights(weights)
