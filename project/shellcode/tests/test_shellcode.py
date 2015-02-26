import unittest
import shellcode as sc
import numpy as np
import numpy.testing as nptest

class TestSpStates(unittest.TestCase):
    """Tests for sp_states function"""

    def test_num_states(self):
        for p in range(1, 20):
            st = list(sc.sp_states(p, 0.5))
            self.assertEqual(len(st), 2*p)

    def test_first_state(self):
        """States should start at p=1"""
        st = list(sc.sp_states(4, 0.5))
        self.assertEqual(st[0][0], 1)

    def test_four_states(self):
        st = np.array(list(sc.sp_states(2, 0.5)))
        exp = np.array([[1, -0.5],
                        [1,  0.5],
                        [2, -0.5],
                        [2,  0.5]])
        nptest.assert_equal(st, exp)


class TestSlater(unittest.TestCase):
    """Tests for the slater function"""

    def test_2part_2level_nstates_paired(self):
        sl = sc.slater(2, 2, 0, pairs_only=True)
        self.assertEqual(len(sl), 2)

    def test_4part_4level_nstates_paired(self):
        sl = sc.slater(4, 4, 0, pairs_only=True)
        self.assertEqual(len(sl), 6)


class TestSDdelta(unittest.TestCase):

    def test_delta(self):
        a = np.arange(5) + 1  # would be [1, 2, 3, 4, 5]

        for delta in range(5):
            b = np.copy(a)
            for i in range(delta):
                b[i] = 0  # not in a
            diff = sc.sd_delta(a, b)
            self.assertEqual(diff, delta)