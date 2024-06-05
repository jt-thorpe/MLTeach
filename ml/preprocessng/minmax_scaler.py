import numpy as np


class MinMaxScaling:
    """Min-max scaling of data.

    Min-max scaling is a method of scaling each feature of the data to have a minimum of 0 and a maximum of 1.

    X* = (X - min(X)) / (max(X) - min(X))

        where:
            X* is the scaled feature (column) of the data
            X is the data feature (column)
            min(X) is the minimum value of the data feature (column)
            max(X) is the maximum value of the data feature (column)

    Attributes:
        range_: Range of the scaled data.
        range_min_: Minimum of the range of the scaled data.
        range_max_: Maximum of the range of the scaled data.
        min_: Minimum of each feature of the data.
        max_: Maximum of each feature of the data.
        scale_factor_: Scale factor for each feature of the data.
    """

    def __init__(self, scaling_range: tuple = (0, 1)) -> None:
        """Initialise the min-max scaling object."""
        self.range_: tuple = scaling_range
        self.range_min_: float = scaling_range[0]
        self.range_max_: float = scaling_range[1]
        self.min_: np.array = np.array([])
        self.max_: np.array = np.array([])
        self.scale_factor_: np.array = np.array([])

    def fit(self, X: np.array) -> None:
        """Compute the minimum and maximum for each feature of the data, and the scale factor.

        Args:
            X: Matrix of data.
        """
        if X.size == 0:
            raise ValueError("Error: The data matrix is empty.")

        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.scale_factor_ = (self.range_max_ - self.range_min_) / (self.max_ - self.min_)

    def transform(self, X: np.array) -> np.array:
        """Scale the data.

        Args:
            X: Matrix of data.

        Returns:
            Scaled matrix of data.
        """
        if self.min_.size == 0 or self.max_.size == 0 or self.scale_factor_.size == 0:
            raise ValueError("Error: It appears the data has not been fitted. Try calling the fit method first.")

        X_scaled = (X - self.min_) * self.scale_factor_ + self.min_range_
        return X_scaled

    def fit_transform(self, X: np.array) -> np.array:
        """Compute the minimum and maximum for each feature of the data, and the scale factor, and scale the data.

        Args:
            X: Matrix of data.

        Returns:
            Scaled matrix of data.
        """
        self.fit(X)
        return self.transform(X)
