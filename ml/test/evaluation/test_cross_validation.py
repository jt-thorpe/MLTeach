import unittest

import numpy as np

from ...evaluation.cross_validation import KFold


class TestKFold(unittest.TestCase):
    """Tests for the KFold class."""

    def setUp(self):
        """Set up the test fixtures."""
        self.kfold = KFold(n_splits=5)

    def test_init(self):
        """Test that the model is initialized with the correct attributes."""
        self.assertIsInstance(self.kfold, KFold)
        self.assertEqual(self.kfold.n_splits, 5)

    def test_split(self):
        """Test that the model splits the data into k-folds."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([1, 2, 3, 4, 5])

        train_indices, test_indices = next(self.kfold.split(X, y))

        self.assertIsInstance(train_indices, np.ndarray)
        self.assertIsInstance(test_indices, np.ndarray)
        self.assertEqual(train_indices.shape, (4,))
        self.assertEqual(test_indices.shape, (1,))

    def test_split_fixed_example(self):
        """Test that the model splits the data into k-folds.

        The calculation is as follows:
            X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
            y = [1, 2, 3, 4, 5]

            n_samples = 5
            fold_size = 5 // 5 = 1
            indices = [0, 1, 2, 3, 4]

            i = 0
                start = 0 * 1 = 0
                end = (0 + 1) * 1 = 1
                test_indices = [0]
                train_indices = [1, 2, 3, 4]
            i = 1
                start = 1 * 1 = 1
                end = (1 + 1) * 1 = 2
                test_indices = [1]
                train_indices = [0, 2, 3, 4]
            i = 2
                start = 2 * 1 = 2
                end = (2 + 1) * 1 = 3
                test_indices = [2]
                train_indices = [0, 1, 3, 4]
            i = 3
                start = 3 * 1 = 3
                end = (3 + 1) * 1 = 4
                test_indices = [3]
                train_indices = [0, 1, 2, 4]
            i = 4
                start = 4 * 1 = 4
                end = (4 + 1) * 1 = 5
                test_indices = [4]
                train_indices = [0, 1, 2, 3]
        """
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([1, 2, 3, 4, 5])

        indices = list(self.kfold.split(X, y))

        self.assertEqual(len(indices), 5)
        self.assertEqual(indices[0][0].tolist(), [1, 2, 3, 4])
        self.assertEqual(indices[0][1].tolist(), [0])
        self.assertEqual(indices[1][0].tolist(), [0, 2, 3, 4])
        self.assertEqual(indices[1][1].tolist(), [1])
        self.assertEqual(indices[2][0].tolist(), [0, 1, 3, 4])
        self.assertEqual(indices[2][1].tolist(), [2])
        self.assertEqual(indices[3][0].tolist(), [0, 1, 2, 4])
        self.assertEqual(indices[3][1].tolist(), [3])
        self.assertEqual(indices[4][0].tolist(), [0, 1, 2, 3])
        self.assertEqual(indices[4][1].tolist(), [4])


if __name__ == '__main__':
    unittest.main()
