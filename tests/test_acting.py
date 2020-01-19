import unittest

import numpy as np

from pyreinforce.acting import EpsGreedyPolicy, DecayingEpsGreedyPolicy,\
    CustomEpsGreedyPolicy, SoftmaxPolicy, OrnsteinUhlenbeckPolicy


class EpsGreedyPolicyTest(unittest.TestCase):
    def setUp(self):
        self._eps = 0.1
        self._acting = EpsGreedyPolicy(self._eps)

    def test_seed(self):
        seed = 123
        eps = 0.7
        lowest_q = 1
        highest_q = 10
        n_actions = 10
        n_qs = 1000

        acting1 = EpsGreedyPolicy(eps)
        acting1.seed(seed)

        acting2 = EpsGreedyPolicy(eps)
        acting2.seed(seed)

        qs = [np.random.uniform(lowest_q, highest_q, size=(1, n_actions)) for _ in range(n_qs)]

        for q in qs:
            a1 = acting1.act(q)
            a2 = acting2.act(q)

            self.assertEqual(a1, a2)

    def test_act(self):
        n_total = 10000
        lowest_q = 1
        highest_q = 10
        n_actions = 100

        n_max_q = 0
        n_random = 0

        for _ in range(n_total):
            q = np.random.uniform(lowest_q, highest_q, size=(1, n_actions))
            arg_max = np.argmax(q)
            action = self._acting.act(q)

            if arg_max == action:
                n_max_q += 1
            else:
                n_random += 1

        actual = n_random / n_total

        max_deviation = 0.1
        actual_deviation = abs((self._eps - actual) / self._eps)

        self.assertLess(actual_deviation, max_deviation)


class DecayingEpsGreedyPolicyTest(unittest.TestCase):
    def setUp(self):
        self._eps_start = 1.0
        self._eps_end = 0.0
        self._eps_decay = 2
        self._acting = DecayingEpsGreedyPolicy(self._eps_start, self._eps_end,
                                               self._eps_decay)

    def test_act(self):
        n_total = 10000
        lowest_q = 1
        highest_q = 10
        n_actions = 100
        n_episodes = 100
        max_deviation = 0.1

        for episode in [0, 50, 100]:
            n_max_q = 0
            n_random = 0

            for _ in range(n_total):
                q = np.random.uniform(lowest_q, highest_q, size=(1, n_actions))
                arg_max = np.argmax(q)
                action = self._acting.act(q, cur_step=0, cur_episode=episode,
                                          n_episodes=n_episodes)

                if arg_max == action:
                    n_max_q += 1
                else:
                    n_random += 1

            actual = n_random / n_total

            if self._acting._eps == 0:
                self.assertEqual(0, actual)
            else:
                actual_deviation = abs((self._acting._eps - actual) / self._acting._eps)
                self.assertLess(actual_deviation, max_deviation)


class CustomEpsGreedyPolicyTest(unittest.TestCase):
    def setUp(self):
        def get_eps(**kwargs):
            global_step = kwargs['global_step']

            if global_step < 10:
                return 0.9
            elif global_step < 100:
                return 0.5
            else:
                return 0.1

        self._acting = CustomEpsGreedyPolicy(get_eps)

    def test_act(self):
        n_total = 10000
        lowest_q = 1
        highest_q = 10
        n_actions = 100
        n_episodes = 100
        max_deviation = 0.1

        for global_step in [9, 99, 999]:
            n_max_q = 0
            n_random = 0

            for _ in range(n_total):
                q = np.random.uniform(lowest_q, highest_q, size=(1, n_actions))
                arg_max = np.argmax(q)
                action = self._acting.act(q, cur_step=0, cur_episode=0, n_episodes=n_episodes,
                                          global_step=global_step)

                if arg_max == action:
                    n_max_q += 1
                else:
                    n_random += 1

            actual = n_random / n_total

            if self._acting._eps == 0:
                self.assertEqual(0, actual)
            else:
                actual_deviation = abs((self._acting._eps - actual) / self._acting._eps)
                self.assertLess(actual_deviation, max_deviation)


class SoftmaxPolicyTest(unittest.TestCase):
    def setUp(self):
        self._acting = SoftmaxPolicy()

    def test_act(self):
        n_total = 10000
        max_deviation = 0.1

        probs = np.array([[0.1, 0.2, 0.3, 0.4]])
        self.assertEqual(1, np.sum(probs))

        actions = {}

        for i in range(probs.shape[1]):
            actions[i] = 0

        for _ in range(n_total):
            a = self._acting.act(probs)
            actions[a] += 1

        for a, n_occurrences in actions.items():
            expected_prob = probs[0, a]
            actual_prob = n_occurrences / n_total
            actual_deviation = abs((expected_prob - actual_prob) / expected_prob)
            self.assertLess(actual_deviation, max_deviation)


class OrnsteinUhlenbeckPolicyTest(unittest.TestCase):
    def setUp(self):
        self._shape = 1
        self._mu = 0.0
        self._theta = 0.05
        self._sigma = 0.01
        self._acting = OrnsteinUhlenbeckPolicy(self._shape, self._mu,
                                               self._theta, self._sigma)

    def test_act(self):
        n_total = 1000
        signal = 1

        a = [self._acting.act([signal], cur_step=i) for i in range(n_total)]
        a = np.array(a)
        a -= signal
        a -= self._mu

        mean_a = np.mean(a)
        mean_a = abs(mean_a)
        max_deviation = 0.05
        self.assertLess(mean_a, max_deviation)

        std_a = np.std(a)
        self.assertGreater(std_a, 0)


if __name__ == '__main__':
    unittest.main()
