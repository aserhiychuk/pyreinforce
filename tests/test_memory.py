import unittest
import random

from pyreinforce.memory import Memory, EpisodicMemory


_seed = 123

_random = random.Random()
_random.seed(_seed)

def _random_experience(s=None, a=None, r=None, s1=None, s1_mask=None):
    s = s or _random.random()
    a = a or _random.randint(0, 10)
    r = r or _random.random()
    s1 = s1 or _random.random()
    s1_mask = s1_mask or _random.randint(0, 1)

    return (s, a, r, s1, s1_mask)


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

        batch1 = memory1.sample()
        batch2 = memory2.sample()

        self.assertListEqual(batch1, batch2)

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
        experience = _random_experience()
        self._memory.add(experience)

        actual = self._memory.sample()
        self.assertListEqual([experience], actual)

    def test_sample_ignores_buffer(self):
        experience = _random_experience()
        self._memory.add(experience, buffer=True)

        actual = self._memory.sample()
        self.assertListEqual([], actual)

    def test_sample_honors_batch_size(self):
        for _ in range(self._capacity):
            self._memory.add(_random_experience())

        batch = self._memory.sample()
        self.assertEqual(self._batch_size, len(batch))

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
        self._capacity = 100
        self._batch_size = 8
        self._n_time_steps = 4
        self._memory = EpisodicMemory(self._capacity, self._batch_size, self._n_time_steps)

    def test_add(self):
        experience = _random_experience()
        self._memory.add(experience)

        self.assertEqual(0, len(self._memory._samples))

        self.assertEqual(1, len(self._memory._buffer))
        self.assertIn(experience, self._memory._buffer)

    def test_sample(self):
        for episode_no in range(_random.randint(10, 20)):
            for step_no in range(_random.randint(10, 40)):
                self._memory.add(_random_experience(s=(episode_no, step_no)))

            self._memory.flush()

        batch = self._memory.sample()
        self.assertEqual(self._batch_size, len(batch))
        
        for episode in batch:
            # s = (episode_no, step_no)
            states = [s for s, _, _, _, _ in episode]
            self.assertEqual(self._n_time_steps, len(states))

            episode_numbers = [episode_no for episode_no, _ in states]
            is_same_episode = all(episode_no == episode_numbers[0] for episode_no in episode_numbers)
            self.assertTrue(is_same_episode)

            step_numbers = [step_no for _, step_no in states]
            is_consecutive_steps = all(step_numbers[0] == (step_no - index) for index, step_no in enumerate(step_numbers))
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
