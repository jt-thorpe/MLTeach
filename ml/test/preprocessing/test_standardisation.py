"""Tests for the Standardisation class.

Sklearn's StandardScaler is used to compare the results of the Standardisation class.
"""

import unittest

import numpy as np
from hypothesis import given
from sklearn.discriminant_analysis import StandardScaler

from mlteach.ml.preprocessng.standardisation import Standardisation
from mlteach.ml.test.strategies.matrix_strategy import matrix_gen


class TestStandardisation(unittest.TestCase):
    """Tests for the Standardisation class."""

    def setUp(self):
        """Set up the test fixtures."""
        self.standardisation = Standardisation()

    def test_get_mean_raises_error_if_data_has_not_been_fitted(self):
        """Test that the get_mean method raises a ValueError if the data has not been fitted."""
        with self.assertRaises(ValueError):
            self.standardisation.get_mean()

    def test_get_std_raises_error_if_data_has_not_been_fitted(self):
        """Test that the get_std method raises a ValueError if the data has not been fitted."""
        with self.assertRaises(ValueError):
            self.standardisation.get_std()

    def test_fit_raises_error_if_data_matrix_is_empty(self):
        """Test that the fit method raises a ValueError if the data matrix is empty."""
        with self.assertRaises(ValueError):
            self.standardisation.fit(np.array([]))

    def test_fit_raises_error_if_data_matrix_contains_nans(self):
        """Test that the fit method raises a ValueError if the data matrix contains NaNs."""
        with self.assertRaises(ValueError):
            self.standardisation.fit(np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]]))

    def test_fit_raises_error_if_data_matrix_contains_infinities(self):
        """Test that the fit method raises a ValueError if the data matrix contains infinities."""
        with self.assertRaises(ValueError):
            self.standardisation.fit(np.array([[1, 2, 3], [4, np.inf, 6], [7, 8, 9]]))

    @given(X=matrix_gen(min_rows=1, max_rows=10, min_cols=1, max_cols=5, min_value=-100, max_value=100))
    def test_fit_computes_mean(self, X):
        """Test that the fit method computes the mean of the data."""
        assert X.size != 0

        self.standardisation.fit(X)

        sklearn_std = StandardScaler()
        sklearn_std.fit(X)

        np.testing.assert_allclose(self.standardisation.mean, np.mean(X, axis=0), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(self.standardisation.mean, sklearn_std.mean_, rtol=1e-5, atol=1e-5)

    def test_fit_computes_std_when_values_are_zero(self):
        """Test that the fit method computes the standard deviation of the data when the values are zero."""
        X = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.standardisation.fit(X)

        np.testing.assert_allclose(self.standardisation.std, np.array([1e-6, 1e-6, 1e-6]), rtol=1e-6, atol=1e-6)

    @given(X=matrix_gen(min_rows=1, max_rows=10, min_cols=1, max_cols=5, min_value=-100, max_value=100))
    def test_fit_computes_standard_deviation(self, X):
        """Test that the fit method computes the standard deviation of the data.

        Zero and very small standard deviations are avoided by adding a small value to the standard deviation.
        This approach differs from the one used in the StandardScaler class from sklearn, so sklearn is not used as a comparision here.
        """
        assert X.size != 0

        self.standardisation.fit(X)

        np.testing.assert_allclose(self.standardisation.std, np.std(X, axis=0), rtol=1e-6, atol=1e-6)

    def test_transform_raises_error_if_data_has_not_been_fitted(self):
        """Test that the transform method raises a ValueError if the data has not been fitted."""
        with self.assertRaises(ValueError):
            self.standardisation.transform(np.array([]))

    @given(X=matrix_gen(min_rows=1, max_rows=10, min_cols=1, max_cols=5, min_value=-1000, max_value=100))
    def test_transform_standardises_data(self, X):
        """Test that the transform method standardises the data."""
        self.standardisation.fit(X)
        np.testing.assert_allclose(self.standardisation.transform(
            X), (X - self.standardisation.mean) / (self.standardisation.std + 1e-8), rtol=1e-6, atol=1e-6)

    @given(X=matrix_gen(min_rows=1, max_rows=10, min_cols=1, max_cols=5, min_value=-1000, max_value=100))
    def test_fit_transform(self, X):
        """Test that the fit_transform method computes the mean and standard deviation of the data and standardises the data."""
        self.standardisation.fit_transform(X)
        np.testing.assert_allclose(self.standardisation.mean, np.mean(X, axis=0), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(self.standardisation.std, np.std(X, axis=0), rtol=1e-6, atol=1e-6)

    def test_get_mean_raises_error_if_data_has_not_been_fitted(self):
        """Test that the get_mean method raises a ValueError if the data has not been fitted."""
        with self.assertRaises(ValueError):
            self.standardisation.get_mean()

    def test_get_std_raises_error_if_data_has_not_been_fitted(self):
        """Test that the get_std method raises a ValueError if the data has not been fitted."""
        with self.assertRaises(ValueError):
            self.standardisation.get_std()

    @given(X=matrix_gen(min_rows=1, max_rows=10, min_cols=1, max_cols=5, min_value=-1000, max_value=100))
    def test_get_mean(self, X):
        """Test that the get_mean method returns the mean of the data."""
        self.standardisation.fit(X)
        np.testing.assert_allclose(self.standardisation.get_mean(), np.mean(X, axis=0), rtol=1e-6, atol=1e-6)

    @given(X=matrix_gen(min_rows=1, max_rows=10, min_cols=1, max_cols=5, min_value=-1000, max_value=100))
    def test_get_std(self, X):
        """Test that the get_std method returns the standard deviation of the data."""
        self.standardisation.fit(X)
        np.testing.assert_allclose(self.standardisation.get_std(), np.std(X, axis=0), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
