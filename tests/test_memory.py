import unittest
import random

import numpy as np

from pyreinforce.memory import Memory, EpisodicMemory


_seed = 123

_random = random.Random()
_random.seed(_seed)

def _random_experience(s=None, a=None, r=None, s1=None, is_terminal=None):
    s = s or _random.random()
    a = a or _random.randint(0, 10)
    r = r or _random.random()
    s1 = s1 or _random.random()
    is_terminal = is_terminal or bool(_random.randint(0, 1))

    return (s, a, r, s1, is_terminal)


class MemoryTest(unittest.TestCase):
    def setUp(self):
        self._capacity = 5
        self._batch_size = 2
        self._memory = Memory(self._capacity, self._batch_size)

    def test_seed(self):
        capacity = 100
        batch_size = 16
        n_experiences = 120
        seed = 123

        memory1 = Memory(capacity, batch_size)
        memory1.seed(seed)

        memory2 = Memory(capacity, batch_size)
        memory2.seed(seed)

        for _ in range(n_experiences):
            experience = _random_experience()

            memory1.add(experience)
            memory2.add(experience)

        # (s, a, r, s1, s1_mask)
        batch1 = memory1.sample()
        batch2 = memory2.sample()

        for b1, b2 in zip(batch1, batch2):
            self.assertTrue(np.array_equal(b1, b2))

    def test_add_honors_capacity(self):
        n_experiences = self._capacity + 3
        experiences = [_random_experience() for _ in range(n_experiences)]

        for experience in experiences:
            self._memory.add(experience)

        self.assertEqual(self._capacity, len(self._memory._samples))

        expected_samples = experiences[n_experiences - self._capacity:]
        self.assertListEqual(expected_samples, self._memory._samples)

    def test_add_to_memory(self):
        experience = _random_experience()
        self._memory.add(experience)

        self.assertEqual(1, len(self._memory._samples))
        self.assertIn(experience, self._memory._samples)

        self.assertEqual(0, len(self._memory._buffer))

    def test_add_to_buffer(self):
        experience = _random_experience()
        self._memory.add(experience, buffer=True)

        self.assertEqual(0, len(self._memory._samples))

        self.assertEqual(1, len(self._memory._buffer))
        self.assertIn(experience, self._memory._buffer)

    def test_sample_via_memory(self):
        experience1 = _random_experience()
        experience2 = _random_experience()
        self._memory.add(experience1)
        self._memory.add(experience2)

        s, a, r, s1, s1_mask = self._memory.sample()

        self.assertEqual((2,), s.shape)
        self.assertEqual((2,), a.shape)
        self.assertEqual((2,), r.shape)
        self.assertEqual((2,), s1.shape)
        self.assertEqual((2,), s1_mask.shape)

    def test_sample_ignores_buffer(self):
        for _ in range(5 * self._batch_size):
            experience = _random_experience()
            self._memory.add(experience, buffer=True)

        actual = self._memory.sample()
        self.assertIsNone(actual)

    def test_sample_honors_batch_size(self):
        for _ in range(self._capacity):
            self._memory.add(_random_experience())

        # (s, a, r, s1, s1_mask)
        batch = self._memory.sample()

        for b in batch:
            self.assertEqual(self._batch_size, b.shape[0])

    def test_flush_no_disount(self):
        experience = _random_experience()
        self._memory.add(experience, buffer=True)

        self._memory.flush()

        self.assertEqual(1, len(self._memory._samples))
        self.assertIn(experience, self._memory._samples)

        self.assertEqual(0, len(self._memory._buffer))

    def test_flush_with_disount(self):
        for _ in range(3):
            experience = _random_experience(r=1)
            self._memory.add(experience, buffer=True)

        self._memory.flush(gamma=0.9)
        
        self.assertAlmostEqual(1, self._memory._samples[2][2], places=4)
        self.assertAlmostEqual(1.9, self._memory._samples[1][2])
        self.assertAlmostEqual(2.71, self._memory._samples[0][2])


class EpisodicMemoryTest(unittest.TestCase):
    def setUp(self):
        self._capacity = 1000
        self._batch_size = 8
        self._n_time_steps = 4
        self._memory = EpisodicMemory(self._capacity, self._batch_size, self._n_time_steps)

    def test_add_honors_capacity(self):
        for episode_no in range(100):
            for step_no in range(_random.randint(10, 100)):
                experience = _random_experience(s=(episode_no, step_no))
                self._memory.add(experience)

            self._memory.flush()

            expected_size = sum([len(episode) for episode in self._memory._samples])
            self.assertEqual(expected_size, self._memory._size)
            self.assertGreaterEqual(self._capacity, self._memory._size)

    def test_add_to_buffer(self):
        experience = _random_experience()
        self._memory.add(experience)

        self.assertEqual(0, len(self._memory._samples))

        self.assertEqual(1, len(self._memory._buffer))
        self.assertIn(experience, self._memory._buffer)

    def test_sample(self):
        for episode_no in range(_random.randint(10, 20)):
            for step_no in range(_random.randint(10, 40)):
                self._memory.add(_random_experience(s=(episode_no, step_no),
                                                    s1=(episode_no, step_no + 1)))

            self._memory.flush()

        s, a, r, s1, s1_mask = self._memory.sample()

        # Sample states:
        #
        # s = [episode_no, step_no]
        #
        # [[[ 5  7] [ 5  8] [ 5  9] [ 5 10]]
        #  [[ 4  8] [ 4  9] [ 4 10] [ 4 11]]
        #  [[18 11] [18 12] [18 13] [18 14]]
        #  [[16 18] [16 19] [16 20] [16 21]]
        #  [[ 8 25] [ 8 26] [ 8 27] [ 8 28]]
        #  [[10 24] [10 25] [10 26] [10 27]]
        #  [[ 7  0] [ 7  1] [ 7  2] [ 7  3]]
        #  [[13  1] [13  2] [13  3] [13  4]]]
        self.assertEqual((self._batch_size, self._n_time_steps, 2), s.shape)
        self.assertEqual((self._batch_size,), a.shape)
        self.assertEqual((self._batch_size,), r.shape)
        self.assertEqual((self._batch_size, self._n_time_steps, 2), s1.shape)
        self.assertEqual((self._batch_size,), s1_mask.shape)

        # Sample episode numbers:
        # [[ 5  5  5  5]
        #  [ 4  4  4  4]
        #  [18 18 18 18]
        #  [16 16 16 16]
        #  [ 8  8  8  8]
        #  [10 10 10 10]
        #  [ 7  7  7  7]
        #  [13 13 13 13]]
        episode_numbers = s[:, :, 0]
        init_episode_no = episode_numbers[:, 0]
        init_episode_no = np.reshape(init_episode_no, (-1, 1))
        is_same_episode = np.all(episode_numbers == init_episode_no)
        self.assertTrue(is_same_episode)

        # Sample step numbers:
        # [[ 7  8  9 10]
        #  [ 8  9 10 11]
        #  [11 12 13 14]
        #  [18 19 20 21]
        #  [25 26 27 28]
        #  [24 25 26 27]
        #  [ 0  1  2  3]
        #  [ 1  2  3  4]]
        step_numbers = s[:, :, 1]
        is_consecutive_steps = np.all(np.diff(step_numbers) == 1)
        self.assertTrue(is_consecutive_steps)

    def test_flush(self):
        episode = [_random_experience() for _ in range(10)]

        for experience in episode:
            self._memory.add(experience)

        self._memory.flush()

        self.assertEqual(1, len(self._memory._samples))
        self.assertListEqual(episode, self._memory._samples[0])


if __name__ == '__main__':
    unittest.main()
