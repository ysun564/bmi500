import random
from numbers import Number

# create a function to compute the dot product of two vectors using a for loop
# add comments for the selected function
def _validate_vector(vector, name):
    """Validate that an input is a non-empty, one-dimensional numeric vector."""
    if not isinstance(vector, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    if not vector:
        raise ValueError(f"{name} must not be empty")

    for value in vector:
        if isinstance(value, (list, tuple)):
            raise ValueError(f"{name} must be one-dimensional")
        if not isinstance(value, Number) or isinstance(value, bool):
            raise TypeError(f"{name} values must be numeric")


def dot_product(vector_a, vector_b):
    """Return the dot product of two equally sized numeric vectors."""
    _validate_vector(vector_a, "vector_a")
    _validate_vector(vector_b, "vector_b")

    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have the same length")

    # Multiply corresponding values and accumulate their sum.
    result = 0
    for index in range(len(vector_a)):
        result += vector_a[index] * vector_b[index]
    return result



# create a function to compute the matrix-vector product using the dot_product function
# add comments for the selected function
def matrix_vector_product(matrix, vector):
    """Return the product of a two-dimensional matrix and a vector."""
    if not isinstance(matrix, (list, tuple)):
        raise TypeError("matrix must be a list or tuple")
    if not matrix:
        raise ValueError("matrix must not be empty")

    for row in matrix:
        if not isinstance(row, (list, tuple)):
            raise ValueError("matrix must be two-dimensional")

    row_length = len(matrix[0])
    if any(len(row) != row_length for row in matrix):
        raise ValueError("matrix rows must have the same length")

    # dot_product validates that each row and the vector contain numbers.
    _validate_vector(vector, "vector")
    if row_length != len(vector):
        raise ValueError("matrix columns must equal the vector length")

    return [dot_product(row, vector) for row in matrix]



# create a main function to test the matrix-vector product function using randomly generated data of size 1000x1000
# add comments for the selected function
def main(size=1000):
    """Generate random square inputs and demonstrate matrix-vector multiplication."""
    # A smaller size may be supplied by tests; the command-line default is 1000.
    matrix = [[random.random() for _ in range(size)] for _ in range(size)]
    vector = [random.random() for _ in range(size)]
    result = matrix_vector_product(matrix, vector)
    print(f"Computed a matrix-vector product with {len(result)} output values.")
    return result


if __name__ == "__main__":
    main()
