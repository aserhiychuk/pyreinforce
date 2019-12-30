import unittest
import random

from pyreinforce.converter import Converter


class TestConverter(Converter):
    pass


class ConverterTest(unittest.TestCase):
    def setUp(self):
        self._converter = TestConverter()

    def test_convert_state(self):
        s = random.random()
        actual = self._converter.convert_state(s)

        self.assertEqual(s, actual)

    def test_convert_action(self):
        a = random.randint(1, 100)
        actual = self._converter.convert_action(a)

        self.assertEqual(a, actual)

    def test_convert_experience(self):
        experience = (random.random(),)
        actual = self._converter.convert_experience(experience)
        self.assertEqual(experience, actual)


if __name__ == '__main__':
    unittest.main()
