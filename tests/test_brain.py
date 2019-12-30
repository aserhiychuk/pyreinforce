import unittest

from pyreinforce.brain import Brain


class TestBrain(Brain):
    pass


class BrainTest(unittest.TestCase):
    def setUp(self):
        self._brain = TestBrain()

    def test_train(self):
        with self.assertRaises(NotImplementedError):
            self._brain.train()


if __name__ == '__main__':
    unittest.main()
