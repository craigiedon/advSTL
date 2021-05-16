import unittest

from stl import *


class TestSTL(unittest.TestCase):

    def test_finally_stl_rob(self):
        spec = F(GEQ0(lambda p: p - 3), 0, 3)
        states = [0.0, 0.0, 3.0, 0.0]
        rob_score = stl_rob(spec, states, 0)
        self.assertEqual(0.0, rob_score)

    def test_smooth_min(self):
        return None

    def test_smooth_max(self):
        return None

    def test_finally_sc_rob(self):
        spec = F(GEQ0(lambda p: p - 3), 0, 3)
        states = [0.0, 0.0, 100.0, 0.0]
        rob_score = sc_rob_pos(spec, states, 0, 100.0)
        self.assertEqual(0.0, rob_score)


if __name__ == '__main__':
    unittest.main()
