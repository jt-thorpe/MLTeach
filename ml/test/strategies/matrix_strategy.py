"""This module contains custom strategies for Hypothesis."""
import numpy as np
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import SearchStrategy, composite, floats, integers


def matrix_gen(min_rows: int, max_rows: int, min_cols: int, max_cols: int, min_value: float, max_value: float) -> SearchStrategy:
    """Defines a strategy for generating random matrices.

    Args:
        min_rows: Minimum number of rows in the matrix.
        max_rows: Maximum number of rows in the
        min_cols: Minimum number of columns in the matrix.
        max_cols: Maximum number of columns in the matrix.
        min_value: Minimum value of the matrix elements.
        max_value: Maximum value of the matrix elements.

    Returns:
        A strategy for generating random matrices.
    """
    rows = integers(min_value=min_rows, max_value=max_rows)
    cols = integers(min_value=min_cols, max_value=max_cols)

    return rows.flatmap(
        lambda r: cols.flatmap(
            lambda c: arrays(
                np.float32,
                shape=(r, c),
                elements=floats(min_value, max_value, allow_nan=False,
                                allow_infinity=False, allow_subnormal=False, width=32)
            )
        )
    )


@composite
def matrix_vector_gen(draw, min_rows: int, max_rows: int, min_cols: int, max_cols: int, min_value: float, max_value: float) -> tuple[np.ndarray, np.ndarray]:
    """Defines a strategy for generating random matrices and a vector.

    Generates a NxM matrix and a matching Mx1 vector. Intended for testing the RidgeRegressor model.

    Args:
        min_rows: Minimum number of rows in the matrix.
        max_rows: Maximum number of rows in the
        min_cols: Minimum number of columns in the matrix.
        max_cols: Maximum number of columns in the matrix.
        min_value: Minimum value of the matrix elements.
        max_value: Maximum value of the matrix elements.

    Returns:
        A strategy for generating random matrices and a vector.
    """
    matrix = draw(matrix_gen(min_rows, max_rows, min_cols, max_cols, min_value, max_value))
    vector = draw(arrays(np.float32, shape=(matrix.shape[0],), elements=floats(
        min_value, max_value, allow_nan=False, allow_infinity=False, allow_subnormal=False, width=32)))

    return matrix, vector
