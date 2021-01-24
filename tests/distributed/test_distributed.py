import unittest
import random
from multiprocessing import shared_memory

import numpy as np

from pyreinforce.distributed import DistributedAgent, WorkerAgent, SharedWeights

from tests import TestEnv as DummyEnv
from tests.distributed import DummyBrain, DummyConn, DummySharedWeights


class DistributedDummyAgent(DistributedAgent):
    def __init__(self, n_episodes, env, brain, train_freq, validation_freq=None,
                 validation_episodes=None, converter=None, callback=None, n_workers=None):
        super().__init__(n_episodes, env, brain, train_freq, validation_freq, validation_episodes,
                         converter, callback, n_workers)

    def _create_worker(self, worker_no, conn_to_parent, shared_weights, validation_barrier, env, brain, *args, **kwargs):
        validation_episodes = self._validation_episodes[worker_no] if self._validation_freq else None

        worker = DummyWorker(worker_no, conn_to_parent, shared_weights, validation_barrier,
                             self._n_episodes, env, brain, self._train_freq,
                             self._validation_freq, validation_episodes,
                             self._converter, self._callback)

        return worker


class DummyWorker(WorkerAgent):
    def __init__(self, worker_no, conn_to_parent, shared_weights, validation_barrier,
                 n_episodes, env, brain, train_freq, validation_freq=None,
                 validation_episodes=None, converter=None, callback=None):
        super().__init__(worker_no, conn_to_parent, shared_weights, validation_barrier,
                         n_episodes, env, brain, train_freq, validation_freq,
                         validation_episodes, converter, callback)

    def _act(self, s, **kwargs):
        return random.randint(-5, 5)

    def _observe(self, experience):
        super()._observe(experience)

        assert len(self._experience_buffer) < self._train_freq, \
            f'experience buffer length. expected < {self._train_freq}, actual: {len(self._experience_bugger)}'

    def _compute_grads(self, batch):
        s, a, r, s1, s1_mask = batch
        assert s.shape[0] <= self._train_freq, f's batch size. expected <= {self._train_freq}, actual: {s.shape[0]}'
        assert a.shape[0] <= self._train_freq, f'a batch size. expected <= {self._train_freq}, actual: {a.shape[0]}'
        assert r.shape[0] <= self._train_freq, f'r batch size. expected <= {self._train_freq}, actual: {r.shape[0]}'
        assert s1.shape[0] <= self._train_freq, f's1 batch size. expected <= {self._train_freq}, actual: {s1.shape[0]}'
        assert s1_mask.shape[0] <= self._train_freq, f's1_mask batch size. expected <= {self._train_freq}, actual: {s1_mask.shape[0]}'

        return []


def _create_env(worker_no=None):
    return DummyEnv()

def _create_brain(worker_no=None):
    return DummyBrain()


class DistributedAgentTest(unittest.TestCase):
    def setUp(self):
        self._n_episodes = 10
        self._train_freq = 5
        self._converter = None
        self._callback = None
        self._n_workers = 4

    def test_run(self):
        validation_freq = None
        validation_episodes = None
        agent = DistributedDummyAgent(self._n_episodes, _create_env, _create_brain, self._train_freq,
                                      validation_freq, validation_episodes,
                                      self._converter, self._callback, self._n_workers)
        rewards, _ = agent.run()

        self.assertSequenceEqual(range(self._n_workers), sorted(rewards.keys()))

        for _, worker_rewards in rewards.items():
            self.assertEqual(self._n_episodes, len(worker_rewards))


class WorkerAgentTest(unittest.TestCase):
    def setUp(self):
        worker_no = 3
        conn_to_parent = DummyConn(worker_no)
        brain = DummyBrain()
        weights_metadata = [(w.shape, w.dtype, w.nbytes) for w in brain.get_weights()]
        shared_weights = DummySharedWeights(weights_metadata)
        barrier = None
        self._n_episodes = 10
        env = DummyEnv()
        train_freq = 5
        validation_freq = None
        validation_episodes = None
        converter = None

        def callback(cur_episode, reward, **kwargs):
            assert worker_no == kwargs['worker_no']

        self._worker = DummyWorker(worker_no, conn_to_parent, shared_weights, barrier,
                                   self._n_episodes, env, brain, train_freq, validation_freq,
                                   validation_episodes, converter, callback)

    def test_run(self):
        rewards, _ = self._worker.run()

        self.assertEqual(self._n_episodes, len(rewards))


def _random_weights(shapes):
    weights = [np.random.uniform(size=shape) for shape in shapes]
    weights = [w.astype(np.float32) for w in weights]

    return weights


class SharedWeightsTest(unittest.TestCase):
    def setUp(self):
        self._n_workers = 5
        shapes = [(2, 5), (7, 3)]
        self._expected = [_random_weights(shapes) for _ in range(self._n_workers)]
        weights = _random_weights(shapes)
        self._metadata = [(w.shape, w.dtype, w.nbytes) for w in weights]
        weights_size = self._n_workers * sum([nbytes for _, _, nbytes in self._metadata])
        self._shared_memory = shared_memory.SharedMemory(create=True, size=weights_size)
        self._shared_weights = SharedWeights(self._metadata, self._shared_memory)

    def tearDown(self):
        self._shared_memory.close()
        self._shared_memory.unlink()

    def test_read(self):
        offset = 0

        for worker_weights in self._expected:
            for w in worker_weights:
                a = np.ndarray(w.shape, w.dtype, self._shared_memory.buf, offset)
                a[:] = w[:]

                offset += w.nbytes

        for worker_no in range(self._n_workers):
            expected = self._expected[worker_no]
            actual = self._shared_weights.read(worker_no)

            for e, a in zip(expected, actual):
                self.assertTrue(np.array_equal(e, a))

    def test_write(self):
        for worker_no, worker_weights in enumerate(self._expected):
            self._shared_weights.write(worker_no, worker_weights)

        offset = 0

        for expected in self._expected:
            for e, (shape, dtype, nbytes) in zip(expected, self._metadata):
                a = np.ndarray(shape, dtype, self._shared_memory.buf, offset)
                self.assertTrue(np.array_equal(e, a))

                offset += nbytes


if __name__ == '__main__':
    unittest.main()
