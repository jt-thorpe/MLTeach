"""Test suite for the KernelRidgeRegressor class.

Separate test suites are created for each kernel function. Some overlap in 
tests but not an issue.
"""
import unittest

import numpy as np

from ...models.kernel_ridge_regressor import (LinearKernelRidgeRegressor,
                                              PolynomialKernelRidgeRegressor,
                                              RbfKernelRidgeRegressor)


class TestLinearKernelRidgeRegression(unittest.TestCase):
    """Test suite for a linear KernelRidgeRegressor class."""

    def setUp(self) -> None:
        self.krr = LinearKernelRidgeRegressor(lambda_=1.0)

    def test_init(self):
        """Test that the model is initialized with the correct attributes."""
        self.assertIsInstance(self.krr, LinearKernelRidgeRegressor)
        self.assertIsInstance(self.krr.coef_, np.ndarray)
        self.assertIsInstance(self.krr.lambda_, float)

    def test_kernel_func_raises_error(self):
        """Test that the linear kernel raises a ValueError if the input vectors are not the same size."""
        x1 = np.array([])
        x2 = np.array([])
        with self.assertRaises(ValueError):
            self.krr._kernel_func(x1, x2)

        x1 = np.array([1, 2])
        x2 = np.array([3, 4, 5])
        with self.assertRaises(ValueError):
            self.krr._kernel_func(x1, x2)

        x1 = np.array([1, 2, 3, 4, 5])
        x2 = np.array([6, 7, 8, 9])
        with self.assertRaises(ValueError):
            self.krr._kernel_func(x1, x2)

    def test_linear_kernel_computes(self):
        """Test that the linear kernel is computed correctly."""
        x1 = np.array([1, 2])
        x2 = np.array([3, 4])

        # vectors so just dot product
        # 1*3 + 2*4 = 11
        expected = 11
        result = self.krr._kernel_func(x1, x2)
        self.assertEqual(result, expected)

        x1 = np.array([[1, 2], [3, 4]])
        x2 = np.array([[5, 6], [7, 8]])

        # matrices so matrix product
        # x2.T = [[5, 7], [6, 8]]
        # x1 @ x2.T = [[1*5 + 2*6, 1*7 + 2*8], [3*5 + 4*6, 3*7 + 4*8]]
        # = [[17, 23], [39, 53]]
        expected = np.array([[17, 23], [39, 53]])
        result = self.krr._kernel_func(x1, x2)
        np.testing.assert_array_equal(result, expected)


class TestPolynomialKernelRidgeRegression(unittest.TestCase):
    """Test suite for a polynomial KernelRidgeRegressor class."""

    def setUp(self) -> None:
        self.krr = PolynomialKernelRidgeRegressor(lambda_=1.0, degree=3)

    def test_init(self):
        """Test that the model is initialized with the correct attributes."""
        self.assertIsInstance(self.krr, PolynomialKernelRidgeRegressor)
        self.assertIsInstance(self.krr.coef_, np.ndarray)
        self.assertIsInstance(self.krr.lambda_, float)
        self.assertIsInstance(self.krr.degree, int)

    def test_polynomial_kernel_raises_error(self):
        """Test that the polynomial kernel raises a ValueError if the input vectors are not the same size."""
        x1 = np.ndarray([])
        x2 = np.ndarray([])
        with self.assertRaises(ValueError):
            self.krr._kernel_func(x1, x2)

        x1 = np.array([1, 2])
        x2 = np.array([3, 4, 5])
        with self.assertRaises(ValueError):
            self.krr._kernel_func(x1, x2)

        x1 = np.array([1, 2, 3, 4, 5])
        x2 = np.array([6, 7, 8, 9])
        with self.assertRaises(ValueError):
            self.krr._kernel_func(x1, x2)

    def test_polynomial_kernel_computes(self):
        """Test that the polynomial kernel is computed correctly."""
        x1 = np.array([1, 2])
        x2 = np.array([3, 4])

        # vectors, we use vector dot product
        # x1 dot x2 = 1*3 + 2*4 = 11
        # assuming kernel_coef = 1.0
        # (1 + 11)^3 = 1728
        expected = 1728
        result = self.krr._kernel_func(x1, x2)
        np.testing.assert_array_almost_equal(result, expected)

        x1 = np.array([[1, 2], [3, 4]])
        x2 = np.array([[5, 6], [7, 8]])

        # matrices, we use matrix product
        # x2.T = [[5, 7], [6, 8]]
        # x1 @ x2.T = [[1*5 + 2*6, 1*7 + 2*8], [3*5 + 4*6, 3*7 + 4*8]]
        # = [[17, 23], [39, 53]]
        # assuming kernel_coef = 1.0
        # 1 + [[17, 23], [39, 53]] = [[18, 24], [40, 54]]
        # 18^3 = 5832
        # 24^3 = 13824
        # 40^3 = 64000
        # 54^3 = 157464
        expected = np.array([[5832., 13824.], [64000., 157464.]])
        result = self.krr._kernel_func(x1, x2)
        np.testing.assert_array_almost_equal(result, expected)


class TestRBFKernelRidgeRegression(unittest.TestCase):
    """Test suite for KernelRidgeRegressor class."""

    def setUp(self) -> None:
        self.krr = RbfKernelRidgeRegressor(lambda_=1.0, gamma=1.0)

    def test_init(self):
        self.assertIsInstance(self.krr, RbfKernelRidgeRegressor)
        self.assertIsInstance(self.krr.coef_, np.ndarray)
        self.assertIsInstance(self.krr.lambda_, float)
        self.assertIsInstance(self.krr.gamma, float)

    def test_rbf_kernel_raises_error(self):
        """Test that the rbf kernel raises a ValueError if the input vectors are not the same size."""
        x1 = np.ndarray([])
        x2 = np.ndarray([])
        with self.assertRaises(ValueError):
            self.krr._kernel_func(x1, x2)

        x1 = np.array([1, 2])
        x2 = np.array([3, 4, 5])
        with self.assertRaises(ValueError):
            self.krr._kernel_func(x1, x2)

        x1 = np.array([1, 2, 3, 4, 5])
        x2 = np.array([6, 7, 8, 9])
        with self.assertRaises(ValueError):
            self.krr._kernel_func(x1, x2)

    def test_rbf_kernel_computes(self):
        """Test that the rbf kernel is computed correctly."""
        x1 = np.array([[1, 2], [3, 4]])
        x2 = np.array([[5, 6], [7, 8]])

        # first we do np.sum((x1 ** 2)
        # [1, 2] ** 2 = [1, 4]
        # [3, 4] ** 2 = [9, 16]
        # np.sum([1, 4]) = 5
        # np.sum([9, 16]) = 25
        # which gives us [5, 25]
        # then we reshape
        # giving [5, 25] = [[5], [25]] = x1_sq

        # then we do np.sum((x2 ** 2)
        # [5, 6] ** 2 = [25, 36]
        # [7, 8] ** 2 = [49, 64]
        # np.sum([25, 36]) = 61
        # np.sum([49, 64]) = 113
        # which gives us [61, 113] = x2_sq

        # we then do x1_sq + x2_sq = [[5], [25]] + [61, 113]
        # = [[5+61, 5+113], [25+61, 25+113]]
        # = [[66, 118], [86, 138]]
        # then we have x1 @ x2.T = [[1, 2], [3, 4]] @ [[5, 7], [6, 8]]
        # which is [[1*5 + 2*6, 1*7 + 2*8], [3*5 + 4*6, 3*7 + 4*8]]
        # = [[17, 23], [39, 53]]
        # multiplied by 2 = [[34, 46], [78, 106]]
        # Then we have [[66, 118], [86, 138]] - [[34, 46], [78, 106]]
        # = [[32, 72], [8, 32]]

        # then -gamma * [[32, 72], [8, 32]] = [[-32, -72], [-8, -32]]
        # assuming gamma = 1.0
        # then we have np.exp([[-32, -72], [-8, -32]])
        # which is exp of each element
        # K = [[1.2664165x10^-14], [5.3801861x10^-32], [3.3546262x10^-4], [1.2664165x10^-14]]
        result = self.krr._kernel_func(x1, x2)
        expected = np.array([[1.2664165e-14, 5.3801861e-32], [3.3546262e-4, 1.2664165e-14]])
        np.testing.assert_array_almost_equal(result, expected)
