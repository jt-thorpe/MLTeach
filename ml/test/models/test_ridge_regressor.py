import unittest

import numpy as np
from hypothesis import given

from ...models.ridge_regressor import RidgeRegressor
from ...test.strategies.matrix_strategy import matrix_vector_gen


class TestRidgeRegressor(unittest.TestCase):
    """Tests for the RidgeRegressor class."""

    def setUp(self):
        """Set up the test fixtures."""
        self.ridge = RidgeRegressor(lambda_=0.1)  # Default lambda = 1.0

    def test_init(self):
        """Test that the model is initialized with the correct attributes."""
        self.assertIsInstance(self.ridge, RidgeRegressor)
        self.assertIsInstance(self.ridge.coef_, np.ndarray)
        self.assertIsInstance(self.ridge.lambda_, float)

    def test_fit_raises_error_if_X_train_is_empty(self):
        """Test that the fit method raises a ValueError if the training data matrix is empty."""
        with self.assertRaises(ValueError):
            self.ridge.fit(np.array([]), np.array([1, 2, 3]))

    @given(matrix_vector_gen(min_rows=1, max_rows=100, min_cols=1, max_cols=10, min_value=-100.0, max_value=100.0))
    def test_fit(self, matrix_vector):
        """Test that the model fits the training data."""
        X_train, y_train = matrix_vector
        self.ridge.fit(X_train, y_train)

        self.assertIsInstance(self.ridge.coef_, np.ndarray)
        self.assertEqual(self.ridge.coef_.shape, (X_train.shape[1],))

    def test_fit_fixed_example(self):
        """Test that the model fits the training data."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([[1], [2], [3]])
        self.ridge.fit(X_train, y_train)

        X_train_mean = np.mean(X_train, axis=0)  # [[3,4]]
        y_train_mean = np.mean(y_train)  # 2

        np.testing.assert_allclose(X_train_mean, self.ridge.X_train_mean_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(y_train_mean, self.ridge.y_train_mean_, rtol=1e-6, atol=1e-6)

        X_train = X_train - X_train_mean
        y_train = y_train - y_train_mean

        np.testing.assert_almost_equal(X_train, np.array([[-2, -2], [0, 0], [2, 2]]), decimal=6)
        np.testing.assert_almost_equal(y_train, np.array([[-1], [0], [1]]), decimal=6)

        # X_train.T = [[-2, 0, 2], [-2, 0, 2]]
        A = X_train.T @ X_train
        B = self.ridge.lambda_ * np.identity(X_train.shape[1])

        np.testing.assert_almost_equal(A, np.array([[8, 8], [8, 8]]), decimal=6)
        np.testing.assert_almost_equal(B, np.array([[0.1, 0], [0, 0.1]]), decimal=6)

        C = A + B

        np.testing.assert_almost_equal(C, np.array([[8.1, 8], [8, 8.1]]), decimal=6)

        inv_C = np.linalg.inv(C)

        np.testing.assert_almost_equal(inv_C, np.array([[5.031056, -4.968944], [-4.968944, 5.031056]]), decimal=6)

        D = X_train.T @ y_train

        np.testing.assert_almost_equal(D, np.array([[4], [4]]), decimal=6)

        w = inv_C @ D

        np.testing.assert_almost_equal(w, np.array([[0.248448], [0.248448]]), decimal=6)

    def test_predict_fixed_example(self):
        # TODO: Update fixed example to account for centering data
        """Test that the model predicts the labels for the test data."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([[1], [2], [3]])
        X_test = np.array([[7, 8], [9, 10]])

        self.ridge.fit(X_train, y_train)  # fit
        result = self.ridge.predict(X_test)  # predict

        # Check fit and predict
        np.testing.assert_array_almost_equal(self.ridge.coef_, np.array(
            [[0.2484472], [0.2484472]]), decimal=6)  # coefs are same

        # Centre the test data
        # X_test = X_test - np.mean(X_train, axis=0)
        # X_test = [[7, 8], [9, 10]] - [3, 4]
        X_test = X_test - self.ridge.X_train_mean_
        np.testing.assert_array_almost_equal(X_test, np.array([[4, 4], [6, 6]]), decimal=5)

        # Calculate axis intercept
        # y_train_mean = np.mean(y_train)
        # X_train_mean = np.mean(X_train)
        # coefs = [[0.2484472], [0.2484472]]
        # intercept = y_train_mean - (X_train_mean @ coefs)
        # intercept = 2 - ([3, 4] @ [[0.2484472], [0.2484472]])
        # intercept = 2 - 1.7391304
        # intercept = 0.2608696
        intercept = self.ridge.y_train_mean_ - (self.ridge.X_train_mean_ @ self.ridge.coef_)  # from model
        np.testing.assert_almost_equal(intercept, 0.2608696, decimal=6)  # matches by hand calculation

        # Predict
        # expexted = X_test @ coefs + intercept
        # expected = [[4, 4], [6, 6]] @ [[0.2484472], [0.2484472]] + 0.2608696
        # expected = [[4 * 0.2484472 + 4 * 0.2484472], [6 * 0.2484472 + 6 * 0.2484472]] + 0.2608696
        # expected = [[1.9875776], [2.9813664]] + 0.2608696
        # expected = [[2.2484472], [3.24223602]]
        expected = np.array([[2.2484472], [3.24223602]])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_residual_sum_of_squares(self):
        """Test that the model computes the residual sum of squares."""
        y = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        rss = self.ridge.residual_sum_of_squares(y, y_pred)
        self.assertEqual(rss, 0.0)

    def test_total_sum_of_squares(self):
        """Test that the model computes the total sum of squares."""
        y = np.array([1, 2, 3])
        tss = self.ridge.total_sum_of_squares(y)
        self.assertEqual(tss, 2.0)

    def test_r_squared(self):
        """Test that the model computes the R-squared value."""
        y = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        r_squared = self.ridge.r_squared(y, y_pred)
        self.assertEqual(r_squared, 1.0)

    def test_mean_squared_error(self):
        """Test that the model computes the mean squared error."""
        y = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        mse = self.ridge.mean_squared_error(y, y_pred)
        self.assertEqual(mse, 0.0)

    def test_mean_absolute_error(self):
        """Test that the model computes the mean absolute error."""
        y = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        mae = self.ridge.mean_absolute_error(y, y_pred)
        self.assertEqual(mae, 0.0)
