"""A module for K-Fold Cross Validation."""

import numpy as np
from ml.models.kernel_ridge_regressor import KernelRidgeRegressor
from ml.models.ridge_regressor import RidgeRegressor


class KFold:
    """K-Fold Cross Validation.

    Splits the data into k-folds and iteratively trains and tests the model
    on each fold.

    Attributes:
        n_splits: Number of folds.
    """

    def __init__(self, n_splits: int = 5) -> None:
        """Initialize the model.

        Args:
            n_splits: Number of folds.
        """
        self.n_splits: int = n_splits

    def split(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Split the data into k-folds.

        Args:
            X: Matrix of features.
            y: Vector of labels.

        Returns:
            Array of indices for the training and testing data.
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if len(X) < self.n_splits:
            raise ValueError("n_splits must be less than or equal to the number of samples.")

        n_samples = X.shape[0]
        fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size
            test_indices = indices[start:end]
            train_indices = np.concatenate((indices[:start], indices[end:]))

            yield train_indices, test_indices

    def validate(self, model, X: np.ndarray, y: np.ndarray) -> tuple:
        """Validate the model using k-fold cross validation.

        Args:
            model: Model to validate.
            X: Matrix of features.
            y: Vector of labels.

        Returns:
            Tuple containing the scores for each fold and the average score.
        """
        fold_scores = []

        for train_indices, test_indices in self.split(X, y):
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # Train the model
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2_score = model.r_squared(y_test, y_pred)
            fold_scores.append(r2_score)

        avg_score = np.mean(fold_scores)
        return (fold_scores, avg_score)
