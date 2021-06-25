import unittest
import random
from multiprocessing import shared_memory

import numpy as np

from pyreinforce.core import Callback
from pyreinforce.distributed import DistributedAgent, WorkerAgent,\
    ExperienceBuffer, RecurrentExperienceBuffer, SharedWeights

from tests import TestEnv as DummyEnv
from tests.distributed import DummyBrain, DummyConn, DummySharedWeights


class DistributedDummyAgent(DistributedAgent):
    def _create_worker(self, worker_no, conn_to_parent, shared_weights, validation_barrier,
                       env, brain, *args, **kwargs):
        validation_episodes = self._validation_episodes[worker_no] if self._validation_freq else None

        worker = DummyWorker(worker_no, conn_to_parent, shared_weights, validation_barrier,
                             self._n_episodes, env, brain, self._experience_buffer, self._train_freq,
                             self._validation_freq, validation_episodes,
                             self._converter, self._callback)

        return worker


class DummyWorker(WorkerAgent):
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


class DummyDistributedCallback(Callback):
    def __init__(self, n_episodes, validation_freq, validation_episodes, n_workers):
        self._n_episodes = n_episodes
        self._validation_freq = validation_freq
        self._validation_episodes = validation_episodes
        self._n_workers = n_workers

        self._n_validations = 0

    def _assert_worker_no(self, actual_worker_no):
        assert actual_worker_no <= self._n_workers - 1,\
            f'worker_no. expected <= {self._n_workers - 1}, actual: {actual_worker_no}'

    def on_before_run(self, **kwargs):
        self._assert_worker_no(kwargs['worker_no'])

    def on_after_run(self, **kwargs):
        self._assert_worker_no(kwargs['worker_no'])

        if self._validation_freq is not None:
            expected_n_validations = self._n_episodes // self._validation_freq
            assert expected_n_validations == self._n_validations,\
                f'# of validations. expected: {expected_n_validations}, actual: {self._n_validations}'

    def on_state_change(self, s, **kwargs):
        self._assert_worker_no(kwargs['worker_no'])

    def on_before_episode(self, episode_no, **kwargs):
        self._assert_worker_no(kwargs['worker_no'])

    def on_after_episode(self, episode_no, reward, **kwargs):
        self._assert_worker_no(kwargs['worker_no'])

    def on_before_validation(self, **kwargs):
        if self._validation_freq is None:
            self._test_case.fail('Validation is disabled')

        self._assert_worker_no(kwargs['worker_no'])

        self._n_validations += 1

    def on_after_validation(self, rewards, **kwargs):
        if self._validation_freq is None:
            self._test_case.fail('Validation is disabled')

        self._assert_worker_no(kwargs['worker_no'])
        # validation load should be distributed between workers
        assert len(rewards) <= self._validation_episodes,\
            f'# of validation episodes. expected <= {self._validation_episodes}, actual: {len(rewards)}'


def _create_env(worker_no=None):
    return DummyEnv()

def _create_brain(worker_no=None):
    return DummyBrain()


class DistributedAgentTest(unittest.TestCase):
    def setUp(self):
        self._n_episodes = 10
        self._train_freq = 5
        self._converter = None
        self._n_workers = 4

    def test_run_without_validation(self):
        experience_buffer = ExperienceBuffer()
        validation_freq = None
        validation_episodes = None
        callback = DummyDistributedCallback(self._n_episodes, validation_freq, validation_episodes, self._n_workers)

        agent = DistributedDummyAgent(self._n_episodes, _create_env, _create_brain,
                                      experience_buffer, self._train_freq,
                                      validation_freq, validation_episodes,
                                      self._converter, callback, self._n_workers)
        rewards, _ = agent.run()

        self.assertSequenceEqual(range(self._n_workers), sorted(rewards.keys()))

        for _, worker_rewards in rewards.items():
            self.assertEqual(self._n_episodes, len(worker_rewards))

    def test_run_with_validation(self):
        experience_buffer = ExperienceBuffer()
        validation_freq = 2
        validation_episodes = 3
        callback = DummyDistributedCallback(self._n_episodes, validation_freq, validation_episodes, self._n_workers)

        agent = DistributedDummyAgent(self._n_episodes, _create_env, _create_brain,
                                      experience_buffer, self._train_freq,
                                      validation_freq, validation_episodes,
                                      self._converter, callback, self._n_workers)
        rewards, _ = agent.run()
        rewards = np.array(rewards)

        n_validations = self._n_episodes // validation_freq
        self.assertSequenceEqual([n_validations, validation_episodes], rewards.shape)


class DummyWorkerCallback(Callback):
    def __init__(self, test_case, expected_worker_no):
        self._test_case = test_case
        self._expected_worker_no = expected_worker_no

    def _assert_worker_no(self, actual_worker_no):
        self._test_case.assertEqual(self._expected_worker_no, actual_worker_no)

    def on_before_run(self, **kwargs):
        self._assert_worker_no(kwargs['worker_no'])

    def on_after_run(self, **kwargs):
        self._assert_worker_no(kwargs['worker_no'])

    def on_state_change(self, s, **kwargs):
        self._assert_worker_no(kwargs['worker_no'])

    def on_before_episode(self, episode_no, **kwargs):
        self._assert_worker_no(kwargs['worker_no'])

    def on_after_episode(self, episode_no, reward, **kwargs):
        self._assert_worker_no(kwargs['worker_no'])

    def on_before_validation(self, **kwargs):
        pass

    def on_after_validation(self, rewards, **kwargs):
        pass


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
        experience_buffer = ExperienceBuffer()
        train_freq = 5
        validation_freq = None
        validation_episodes = None
        converter = None
        callback = DummyWorkerCallback(self, worker_no)

        self._worker = DummyWorker(worker_no, conn_to_parent, shared_weights, barrier,
                                   self._n_episodes, env, brain, experience_buffer, train_freq,
                                   validation_freq, validation_episodes,
                                   converter, callback)

    def test_run(self):
        rewards, _ = self._worker.run()

        self.assertEqual(self._n_episodes, len(rewards))


class ExperienceBufferTest(unittest.TestCase):
    def setUp(self):
        self._buffer = ExperienceBuffer()

    def test_add(self):
        experience = (0, 1, 2, 3, 4)
        self._buffer.add(experience)

        self.assertEqual(1, len(self._buffer._buffer))
        self.assertIn(experience, self._buffer._buffer)

    def test_get_batch_and_reset_empty(self):
        batch = self._buffer.get_batch_and_reset()
        self.assertIsNone(batch)

    def test_get_batch_and_reset_non_empty(self):
        n = 17
        expected_s = np.random.uniform(size=(n, 9))
        expected_a = np.random.randint(0, 100, size=(n, 1))
        expected_r = np.random.uniform(size=(n, 1))
        expected_s1 = np.random.uniform(size=(n, 9))
        expected_done = np.random.randint(0, 2, size=(n, 1))

        for i in range(n):
            experience = expected_s[i], expected_a[i, 0], expected_r[i, 0], expected_s1[i], expected_done[i, 0]
            self._buffer.add(experience)

        actual_s, actual_a, actual_r, actual_s1, actual_s1_mask = self._buffer.get_batch_and_reset()
        self.assertEqual(0, len(self._buffer))

        self.assertTrue(np.array_equal(expected_s, actual_s))
        self.assertTrue(np.array_equal(expected_a, actual_a))
        self.assertTrue(np.array_equal(expected_r, actual_r))
        self.assertTrue(np.array_equal(expected_s1, actual_s1))
        self.assertTrue(np.array_equal(1 - expected_done, actual_s1_mask))

    def test_len(self):
        self.assertEqual(0, len(self._buffer))

        n = 17

        for i in range(n):
            experience = (i, 1, 2, i + 1, 0)
            self._buffer.add(experience)
            self.assertEqual(i + 1, len(self._buffer))


class RecurrentExperienceBufferTest(unittest.TestCase):
    def setUp(self):
        self._n_time_steps = 7
        self._buffer = RecurrentExperienceBuffer(self._n_time_steps)

    def test_add(self):
        experience = (0, 1, 2, 3, 4)
        self._buffer.add(experience)

        self.assertEqual(1, len(self._buffer._buffer))
        self.assertIn(experience, self._buffer._buffer)

    def test_get_batch_and_reset_empty(self):
        batch = self._buffer.get_batch_and_reset()
        self.assertIsNone(batch)

    def test_get_batch_and_reset_non_empty(self):
        n = 1000
        s = np.random.uniform(size=(n, 2))
        a = np.random.randint(0, 100, size=(n, 1))
        r = np.random.uniform(size=(n, 1))
        s1 = np.random.uniform(size=(n, 2))
        done = np.random.randint(0, 2, size=(n, 1))

        train_freq = 5

        for n_time_steps in [train_freq - 2, train_freq, train_freq + 2]:
            buffer = RecurrentExperienceBuffer(n_time_steps)

            for i in range(n):
                experience = s[i], a[i, 0], r[i, 0], s1[i], done[i, 0]
                buffer.add(experience)

                is_terminal = np.random.uniform() < 0.025

                if len(buffer) == train_freq or is_terminal:
                    batch = buffer.get_batch_and_reset(is_terminal)
                    self.assertEqual(0, len(buffer))

                    if is_terminal:
                        self.assertEqual(0, len(buffer._buffer))

                    if batch is None:
                        continue

                    actual_s, actual_a, actual_r, actual_s1, actual_s1_mask = batch
                    batch_size = actual_s.shape[0]

                    if is_terminal:
                        self.assertGreaterEqual(train_freq, batch_size)
                    else:
                        self.assertEqual(train_freq, batch_size)

                    expected_s = [s[i - batch_size + 1 - n_time_steps + 1 + j:i - batch_size + 1 + 1 + j] for j in range(batch_size)]
                    expected_s = np.array(expected_s)
                    self.assertTrue(np.array_equal(expected_s, actual_s))
                    expected_a = a[i - batch_size + 1:i + 1, :]
                    self.assertTrue(np.array_equal(expected_a, actual_a))
                    expected_r = r[i - batch_size + 1:i + 1, :]
                    self.assertTrue(np.array_equal(expected_r, actual_r))
                    expected_s1 = [s1[i - batch_size + 1 - n_time_steps + 1 + j:i - batch_size + 1 + 1 + j] for j in range(batch_size)]
                    expected_s1 = np.array(expected_s1)
                    self.assertTrue(np.array_equal(expected_s1, actual_s1))
                    expected_s1_mask = 1 - done[i - batch_size + 1:i + 1, :]
                    self.assertTrue(np.array_equal(expected_s1_mask, actual_s1_mask))

    def test_len(self):
        self.assertEqual(0, len(self._buffer))

        n = 19

        for i in range(n):
            experience = (i, 1, 2, i + 1, 0)
            self._buffer.add(experience)

            expected_len = max(0, i + 2 - self._n_time_steps)
            actual_len = len(self._buffer)
            self.assertEqual(expected_len, actual_len)


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
