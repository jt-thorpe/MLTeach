import numpy as np


class RidgeRegressor:
    """Ridge regression model.

    Implements the closed-form solution for ridge regression.

    w = (X^T X + lambda * I)^(-1) X^T y

        where:
            w is the vector of coefficients
            X is the matrix of features
            y is the vector of labels
            lambda is the regularisation parameter
            I is the identity matrix

    Attributes:
        w: Regression weights.
        lambda_: Regularisation parameter.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        """Initialize the model.

        Args:
            lambda: Regularisation parameter.
        """
        self.coef_: np.ndarray = np.ndarray([])  # w: vector of coefficients
        self.lambda_: float = lambda_
        self.intercept_: float = 0.0
        self.X_train_mean_: np.ndarray = np.ndarray([])  # To centre the data
        self.y_train_mean_: np.ndarray = np.ndarray([])  # To centre the data

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model to the training data.

        Compute w, the matrix of coefficients, using the closed-form.

        w = (X^T X + lambda * I)^(-1) X^T y

        Args:
            X_train: Matrix of training data features.
            y_train: Vector of training data labels.
        """
        if X_train.shape[0] == 0:
            raise ValueError("Error: X_train must not be empty")
        if y_train.size == 0:
            raise ValueError("Error: y_train must not be empty")

        assert X_train.shape[0] == y_train.shape[0], "Error: Rows in X_train must match the number of elements in y_train"

        # Store the mean of the training data to centre the data
        self.X_train_mean_ = np.mean(X_train, axis=0)
        self.y_train_mean_ = np.mean(y_train)

        # Centre the data
        X_train = X_train - self.X_train_mean_
        y_train = y_train - self.y_train_mean_

        self.coef_ = np.linalg.inv(X_train.T @ X_train + self.lambda_ *
                                   np.identity(X_train.shape[1])) @ X_train.T @ y_train

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict the labels for the test data.

        Returns the predicted labels for the test data. The predicted labels are computed
        as the dot product of the test data features and the matrix of coefficients.

        Args:
            X_test: Matrix of test data features.

        Returns:
            Vector of predicted labels.
        """
        if self.coef_.size == 0:
            raise ValueError("Model has not been fitted")

        # Centre the data with the mean of the training data
        X_test = X_test - self.X_train_mean_

        # Calculate axis intercept
        self.intercept_ = self.y_train_mean_ - (self.X_train_mean_ @ self.coef_)

        return X_test @ self.coef_ + self.intercept_

    def residual_sum_of_squares(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the residual sum of squares.

        The residual sum of squares is a measure of the variance in the dependent variable
        that is not explained by the independent variables.

        Args:
            y: Vector of labels.
            y_pred: Vector of predicted labels.

        Returns:
            Residual sum of squares.
        """
        return np.sum((y - y_pred) ** 2)

    def total_sum_of_squares(self, y: np.ndarray) -> float:
        """Compute the total sum of squares.

        The total sum of squares is a measure of the total variance in the dependent variable.

        Args:
            y: Vector of labels.

        Returns:
            Total sum of squares.
        """
        return np.sum((y - np.mean(y)) ** 2)

    def r_squared(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the coefficient of determination (R^2).

        The coefficient of determination is a measure of the proportion of variance in the
        dependent variable that is predictable from the independent variables.

        Args:
            y: Vector of labels.
            y_pred: Vector of predicted labels.

        Returns:
            Coefficient of determination.
        """
        tss = self.total_sum_of_squares(y)
        if tss == 0:
            if np.all(y == y_pred):
                return 1.0  # Perfect prediction
            else:
                return 0.0  # Doesnt predict constant
        return 1 - self.residual_sum_of_squares(y, y_pred) / tss

    def mean_squared_error(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the mean squared error.

        The mean squared error is a measure of the average of the squares of the errors between
        the true and predicted labels.

        Args:
            y: Vector of labels.
            y_pred: Vector of predicted labels.

        Returns:
            Mean squared error.
        """
        return np.mean((y - y_pred) ** 2)

    def mean_absolute_error(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the mean absolute error.

        The mean absolute error is a measure of the average of the absolute errors between the
        true and predicted labels.

        Args:
            y: Vector of labels.
            y_pred: Vector of predicted labels.

        Returns:
            Mean absolute error.
        """
        return np.mean(np.abs(y - y_pred))

    def evaluate(self, y: np.ndarray, y_pred: np.ndarray) -> dict:
        """Evaluate the model.

        Evaluate the model using a range of metrics.

        Args:
            y: Vector of labels.
            y_pred: Vector of predicted labels.

        Returns:
            Dictionary of evaluation metrics.
        """
        return {
            "r_squared": self.r_squared(y, y_pred),
            "mean_squared_error": self.mean_squared_error(y, y_pred),
            "mean_absolute_error": self.mean_absolute_error(y, y_pred)
        }
