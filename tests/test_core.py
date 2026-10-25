import unittest
from fractions import Fraction

from py_lattice_basis_reduction import lll_reduce, dot, LLLReducer


class TestDot(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(dot([1, 2, 3], [4, 5, 6]), 32)

    def test_mismatch(self):
        with self.assertRaises(ValueError):
            dot([1, 2], [1, 2, 3])


class TestConstruction(unittest.TestCase):
    def test_empty_basis_rejected(self):
        with self.assertRaises(ValueError):
            LLLReducer([])

    def test_empty_vector_rejected(self):
        with self.assertRaises(ValueError):
            LLLReducer([[]])

    def test_ragged_basis_rejected(self):
        with self.assertRaises(ValueError):
            LLLReducer([[1, 2], [3]])

    def test_delta_too_small(self):
        with self.assertRaises(ValueError):
            LLLReducer([[1, 0], [0, 1]], delta=Fraction(1, 5))

    def test_delta_at_boundary(self):
        with self.assertRaises(ValueError):
            LLLReducer([[1, 0], [0, 1]], delta=Fraction(1, 4))

    def test_delta_too_large(self):
        with self.assertRaises(ValueError):
            LLLReducer([[1, 0], [0, 1]], delta=2)


class TestLLL(unittest.TestCase):
    def test_already_reduced_identity(self):
        b = [[1, 0], [0, 1]]
        r = lll_reduce(b)
        self.assertEqual(r, [[1, 0], [0, 1]])

    def test_two_dimension_swap(self):
        b = [[1, 1], [0, 1]]
        r = lll_reduce(b)
        # The shorter vector [0,1] should come first after reduction.
        self.assertEqual(dot(r[0], r[0]), 1)
        self.assertEqual(dot(r[1], r[1]), 1)

    def test_input_not_mutated(self):
        original = [[1, 1], [0, 1]]
        snapshot = [list(v) for v in original]
        lll_reduce(original)
        self.assertEqual(original, snapshot)

    def test_large_coefficient_reduction(self):
        b = [[1, 0], [1000000, 1]]
        r = lll_reduce(b)
        # Both vectors have norm 1; order depends on the algorithm.
        norms = sorted(dot(v, v) for v in r)
        self.assertEqual(norms, [1, 1])

    def test_three_dimensional(self):
        b = [[1, 1, 1], [2, 3, 1], [3, 1, 2]]
        r = lll_reduce(b)
        # First vector must be the shortest lattice vector found.
        norms = [dot(v, v) for v in r]
        self.assertLessEqual(norms[0], norms[1])
        self.assertLessEqual(norms[0], norms[2])
        # Determinant must be preserved (up to sign).
        det_orig = (b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
                    - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
                    + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]))
        det_red = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                   - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                   + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
        self.assertEqual(abs(det_orig), abs(det_red))

    def test_linearly_dependent_input(self):
        b = [[1, 0], [2, 0]]
        r = lll_reduce(b)
        # Lattice is 1-dimensional; one vector must be zero.
        zero_count = sum(1 for v in r if all(x == 0 for x in v))
        self.assertEqual(zero_count, 1)
        non_zero = [v for v in r if any(x != 0 for x in v)]
        self.assertEqual(len(non_zero), 1)
        self.assertEqual(dot(non_zero[0], non_zero[0]), 1)

    def test_integers_returned(self):
        b = [[1, 1], [0, 1]]
        r = lll_reduce(b)
        for v in r:
            for x in v:
                self.assertIsInstance(x, int)

    def test_delta_one_allowed(self):
        # delta=1 is at the boundary; should still terminate.
        b = [[1, 1], [0, 1]]
        r = lll_reduce(b, delta=1)
        self.assertEqual(dot(r[0], r[0]), 1)


if __name__ == "__main__":
    unittest.main()
