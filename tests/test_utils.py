import unittest

from pyreinforce.utils import discount_rewards


class UtilsTest(unittest.TestCase):
    def test_discount_rewards(self):
        actual = discount_rewards([1, 1, 1], 0.9)
        actual = actual.tolist()

        for e, a in zip([2.71, 1.9, 1.0], actual):
            self.assertAlmostEqual(e, a, places=4)


if __name__ == '__main__':
    unittest.main()
