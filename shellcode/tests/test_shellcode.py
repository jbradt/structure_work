import unittest
import shellcode as sc
import numpy as np
import numpy.testing as nptest
from itertools import permutations, combinations


class TestSpStates(unittest.TestCase):
    """Tests for sp_states function"""

    def test_num_states(self):
        for p in range(1, 20):
            st = list(sc.sp_states(p, 0.5))
            self.assertEqual(len(st), 2 * p)

    def test_first_state(self):
        """States should start at p=1"""
        st = list(sc.sp_states(4, 0.5))
        self.assertEqual(st[0][0], 1)

    def test_four_states(self):
        st = np.array(list(sc.sp_states(2, 0.5)))
        exp = np.array([[1, -0.5],
                        [1, 0.5],
                        [2, -0.5],
                        [2, 0.5]])
        nptest.assert_equal(st, exp)


class TestSlater(unittest.TestCase):
    """Tests for the slater function"""

    def setUp(self):
        self.sps, self.mel = sc.load_interaction('../usdb.txt')

    def test_2part_0m(self):
        sl = sc.slater(2, self.sps, 0)
        self.assertEqual(len(sl), 14)

    def test_3part_05m(self):
        sl = sc.slater(3, self.sps, 1 / 2)
        self.assertEqual(len(sl), 37)

    def test_4part_0m(self):
        sl = sc.slater(4, self.sps, 0)
        self.assertEqual(len(sl), 81)

    def test_values(self):
        sl = sc.slater(2, self.sps, 3)
        exp = [[1, 11],
               [4, 11],
               [5, 10],
               [9, 11]]
        self.assertListEqual(sl, exp)

    def test_ret_types(self):
        """Check that slater returns a list of lists."""
        sl = sc.slater(2, self.sps, 3)
        self.assertTrue(isinstance(sl, list))
        for item in sl:
            self.assertTrue(isinstance(item, list))


class TestSDdelta(unittest.TestCase):
    def test_delta(self):
        a = np.arange(5) + 1  # would be [1, 2, 3, 4, 5]

        for delta in range(5):
            b = np.copy(a)
            for i in range(delta):
                b[i] = 0  # not in a
            diff = sc.sd_delta(a, b)
            self.assertEqual(diff, delta)


class TestMergeSort(unittest.TestCase):
    """Tests for the merge_sort function"""

    def test_sort(self):
        exp = list(range(6))

        for p in permutations(exp, len(exp)):
            p = list(p)
            inv, res = sc.merge_sort(p)
            self.assertListEqual(res, exp)

    def test_inversions(self):
        e0 = [1, 2, 3, 4]
        e1 = [1, 2, 4, 3]
        e2 = [1, 4, 2, 3]
        e3 = [1, 4, 3, 2]
        e4 = [4, 3, 2, 1]
        e5 = [2, 3, 0, 1, 4, 5]

        cases = [(0, e0), (1, e1), (2, e2), (3, e3), (6, e4), (4, e5)]
        for i, a in cases:
            inv, res = sc.merge_sort(a)
            self.assertEqual(inv, i)


@unittest.skip('out of date')
class TestFindEigenvalues(unittest.TestCase):
    """Tests for the function find_pairing_hamiltonian_eigenvalues"""

    def test_no_pairing(self):
        res = sc.find_pairing_hamiltonian_eigenvalues(4, 4, 0, True, g=0)
        exp = np.array([2., 4., 6., 6., 8., 10.])
        nptest.assert_equal(res, exp)

class TestFindHamiltonianMatrix(unittest.TestCase):

    def setUp(self):
        self.sps, self.mel = sc.load_interaction('../usdb.txt')
        self.sds = sc.slater(2, self.sps, total_m=3)
        self.hmat = sc.find_hamiltonian_matrix(self.sds, self.sps, self.mel)

    def test_square(self):
        sh = self.hmat.shape
        self.assertEqual(len(sh), 2)
        self.assertEqual(sh[0], sh[1])

    def test_abs_symmetric(self):
        """Make sure the matrix is at least sort of symmetric."""
        nptest.assert_allclose(np.abs(self.hmat.T), np.abs(self.hmat))

    def test_hermitian(self):
        nptest.assert_allclose(self.hmat.T, self.hmat)