import numpy as np

EMPTY_WARNING = "Error: x1 and x2 must not be empty"
MATRIX_WARNING = "Error: x1 and x2 must be matrices"
FEATURE_WARNING = "Error: x1 and x2 must have the same number of features"


class KernelRidgeRegressor:
    """Kernel Ridge Regression model.

    Base class for kernel ridge regression models.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        """Initialize the model.

        Args:
            lambda_: Regularisation parameter.
            coef_: Coefficients.
            _X_train: Training data for internal use.
        """
        self.lambda_: float = lambda_
        self.coef_: np.ndarray = np.ndarray([])  # w: vector of coefficients
        self._X_train: np.ndarray = np.ndarray([])  # X_train: matrix of training data features, for internal use

    def _kernel_func(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Kernel function.

        Args:
            x1: Input vector.
            x2: Input vector.

        Returns:
            Kernel value.
        """
        raise NotImplementedError("Subclasses must implement _kernel_func")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model to the training data.

        Args:
            X_train: Matrix of training data features.
            y_train: Vector of training data labels.
        """
        self._reset()  # Clear previous values

        if X_train.shape[0] == 0:
            raise ValueError("Error: X_train must not be empty")
        if y_train.size == 0:
            raise ValueError("Error: y_train must not be empty")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Error: X_train and y_train must have the same number of samples")

        # Reshape if X_train is a vector
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)

        self._X_train = X_train  # Needed for prediction

        K_train = self._kernel_func(X_train, X_train)

        self.coef_ = np.linalg.inv(K_train + self.lambda_
                                   * np.identity(X_train.shape[0])) @ y_train

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict the labels for the test data.

        Args:
            X_test: Matrix of test data features.

        Returns:
            Vector of predicted labels.
        """
        if self.coef_.size == 0:
            raise ValueError("Model has not been fitted")

        # Compute the kernel matrix for the test data
        K_test = self._kernel_func(X_test, self._X_train)

        return K_test @ self.coef_

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

    def _reset(self) -> None:
        """Reset the model.

        Clear the coefficients. Used internally to reset the model before fitting.
        """
        self.coef_ = np.ndarray([])


class LinearKernelRidgeRegressor(KernelRidgeRegressor):
    """Kernel Ridge Regression model with linear kernel.

    This is equivalent to the RidgeRegressor model.

    Attributes:
        lambda_: Regularisation parameter.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        """Initialize the model.

        Args:
            lambda_: Regularisation parameter.
        """
        super().__init__(lambda_=lambda_)

    def _kernel_func(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Linear kernel function.

        Args:
            x1: Input vector.
            x2: Input vector.

        Returns:
            Kernel value or matrix.
        """
        if x1.size == 0 or x2.size == 0:
            raise ValueError(EMPTY_WARNING)

        # If x1 and x2 are vectors, compute the dot product
        if x1.ndim == 1 and x2.ndim == 1:
            return np.dot(x1, x2)

        if x1.ndim != 2 or x2.ndim != 2:
            raise ValueError(MATRIX_WARNING)
        if x1.shape[1] != x2.shape[1]:
            raise ValueError(FEATURE_WARNING)

        # If x1 and x2 are matrices, compute the matrix product
        return x1 @ x2.T


class PolynomialKernelRidgeRegressor(KernelRidgeRegressor):
    """Kernel Ridge Regression model with polynomial kernel.

    Attributes:
        lambda_: Regularisation parameter.
        degree: Polynomial degree.
    """

    def __init__(self, lambda_: float = 1.0, degree: int = 3, kernel_coef: float = 1.0) -> None:
        """Initialize the model.

        Args:
            lambda_: Regularisation parameter.
            degree: Polynomial degree.
        """
        super().__init__(lambda_=lambda_)
        self.degree: int = degree
        self.kernel_coef: float = kernel_coef

    def _kernel_func(self, x1: np.ndarray, x2: np.ndarray):
        """Polynomial kernel function.

        Ref: V Vovk, Chapter 7: Kernel Methods, CS3920/CS5920 Machine Learning

        Args:
            x1: Input vector.
            x2: Input vector.
            degree: Polynomial degree.

        Returns:
            Kernel value.
        """
        if x1.size == 0 or x2.size == 0:
            raise ValueError(EMPTY_WARNING)

        # if vectors, use dot product
        if x1.ndim == 1 and x2.ndim == 1:
            return (self.kernel_coef + np.dot(x1, x2)) ** self.degree

        if x1.ndim != 2 or x2.ndim != 2:
            raise ValueError(MATRIX_WARNING)
        if x1.shape[1] != x2.shape[1]:
            raise ValueError(FEATURE_WARNING)

        return (self.kernel_coef + (x1 @ x2.T)) ** self.degree


class RbfKernelRidgeRegressor(KernelRidgeRegressor):
    """Kernel Ridge Regression model with radial basis function (RBF) kernel.

    Attributes:
        lambda_: Regularisation parameter.
        gamma: Rbf kernel gamma parameter (kernel width).
    """

    def __init__(self, lambda_: float = 1.0, gamma: float = 1.0) -> None:
        """Initialize the model.

        Args:
            lambda_: Regularisation parameter.
            gamma: Rbf kernel gamma parameter (kernel width).
        """
        super().__init__(lambda_=lambda_)
        self.gamma: float = gamma

    def _kernel_func(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Rbf kernel function."""
        if x1.size == 0 or x2.size == 0:
            raise ValueError(EMPTY_WARNING)
        if x1.ndim == 1 and x2.ndim == 1:
            raise ValueError("Error: Rbf kernel function does not support vectors as input currently.")
        if x1.ndim != 2 or x2.ndim != 2:
            raise ValueError(MATRIX_WARNING)
        if x1.shape[1] != x2.shape[1]:
            raise ValueError(FEATURE_WARNING)

        x1_sq = np.sum(x1**2, axis=1).reshape(-1, 1)  # reshape to column vector
        x2_sq = np.sum(x2**2, axis=1)
        sq_dists = x1_sq + x2_sq - 2 * x1 @ x2.T  # euclidean distances between x1 and x2
        K = np.exp(-self.gamma * sq_dists)

        return K
