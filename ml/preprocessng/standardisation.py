import numpy as np

UNFIT_ERROR_MESSAGE = "Error: It appears the data has not been fitted. Try calling the fit method first."


class Standardisation:
    """Standardisation of data.

    Standardisation is a method of scaling data to have a mean of 0 and a standard deviation of 1.
    This is achieved by subtracting the mean from each data point and dividing by the standard deviation.

    Attributes:
        mean: Mean of the data.
        std: Standard deviation of the data.
    """

    def __init__(self) -> None:
        """Initialise the standardisation object."""
        self.mean: np.array = np.array([])
        self.std: np.array = np.array([])

    def get_mean(self) -> np.array:
        """Get the mean for each column of the data.

        Returns:
            Mean of the data.
        """
        if self.mean.size == 0:
            raise ValueError(UNFIT_ERROR_MESSAGE)
        return self.mean

    def get_std(self) -> np.array:
        """Get the standard deviation for each column of the data.

        NB: If a standard deviation is 0 or close to 0, it is likely that the data is constant. Consider removing the feature.

        Returns:
            Standard deviation of the data.
        """
        if self.std.size == 0:
            raise ValueError(UNFIT_ERROR_MESSAGE)
        return self.std

    def fit(self, X: np.array) -> None:
        """Compute the mean and standard deviation of the data, feature-wise.

        Args:
            X: Matrix of data.
        """
        # Handle edge cases
        if X.size == 0:
            raise ValueError("Error: The data matrix is empty.")
        if np.isnan(X).any():
            raise ValueError("Error: The data matrix contains NaNs. Please handle NaNs before calling this method.")
        if np.isinf(X).any():
            raise ValueError(
                "Error: The data matrix contains infinities. Please handle infinities before calling this method.")

        # Feature-wise mean and standard deviation
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X: np.array) -> np.array:
        """Standardise the data.

        A small value is added to the standard deviation to avoid division by zero. Standardisation is done column-wise (feature-wise).

        Args:
            X: Matrix of data.

        Returns:
            Standardised matrix of data.
        """
        if self.std.size == 0 or self.mean.size == 0:
            raise ValueError(UNFIT_ERROR_MESSAGE)

        # Need some logic to handle the case where the standard deviation is zero.

        return (X - self.mean) / (self.std + 1e-8)

    def fit_transform(self, X: np.array) -> np.array:
        """Compute the mean and standard deviation of the data and standardise the data.

        Args:
            X: Matrix of data.

        Returns:
            Standardised matrix of data.
        """
        self.fit(X)
        return self.transform(X)
