"""Tests for the matrix-vector multiplication functions."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import matvec_multiply


class DotProductTests(unittest.TestCase):
    """Verify dot-product results and input validation."""

    def test_computes_dot_product(self):
        self.assertEqual(matvec_multiply.dot_product([1, 2, 3], [4, 5, 6]), 32)

    def test_rejects_vectors_with_different_lengths(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            matvec_multiply.dot_product([1, 2], [3])

    def test_rejects_empty_vectors(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            matvec_multiply.dot_product([], [])

    def test_rejects_non_one_dimensional_vectors(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            matvec_multiply.dot_product([[1, 2]], [3])

    def test_rejects_non_numeric_vector_values(self):
        with self.assertRaisesRegex(TypeError, "numeric"):
            matvec_multiply.dot_product([1, "two"], [3, 4])


class MatrixVectorProductTests(unittest.TestCase):
    """Verify matrix-vector results and input validation."""

    def test_computes_matrix_vector_product(self):
        matrix = [[1, 2, 3], [4, 5, 6]]
        vector = [7, 8, 9]

        self.assertEqual(matvec_multiply.matrix_vector_product(matrix, vector), [50, 122])

    def test_rejects_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "columns"):
            matvec_multiply.matrix_vector_product([[1, 2], [3, 4]], [5])

    def test_rejects_non_two_dimensional_matrix(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            matvec_multiply.matrix_vector_product([1, 2], [3, 4])

    def test_rejects_non_one_dimensional_vector(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            matvec_multiply.matrix_vector_product([[1, 2]], [[3], [4]])

    def test_rejects_empty_matrix(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            matvec_multiply.matrix_vector_product([], [1])

    def test_rejects_empty_vector(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            matvec_multiply.matrix_vector_product([[1]], [])

    def test_rejects_non_numeric_matrix_values(self):
        with self.assertRaisesRegex(TypeError, "numeric"):
            matvec_multiply.matrix_vector_product([[1, "two"]], [3, 4])

    def test_rejects_non_numeric_vector_values(self):
        with self.assertRaisesRegex(TypeError, "numeric"):
            matvec_multiply.matrix_vector_product([[1, 2]], [3, "four"])

    def test_rejects_irregular_matrix_rows(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            matvec_multiply.matrix_vector_product([[1, 2], [3]], [4, 5])


class MainTests(unittest.TestCase):
    """Verify that the demonstration uses the multiplication function."""

    def test_main_generates_square_inputs_and_returns_product(self):
        with patch("matvec_multiply.random.random", return_value=1.0):
            with redirect_stdout(StringIO()):
                result = matvec_multiply.main(size=2)

        self.assertEqual(result, [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()

