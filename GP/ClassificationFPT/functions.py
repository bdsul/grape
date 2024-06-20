import random
import numpy as np
import math
import re

def replace_substrings(input_string, replacements):
    # Function to replace matched pattern with corresponding list value
    def replacer(match):
        index = int(match.group(1))  # Extract the number after 'IN'
        return replacements[index]   # Return the corresponding list value

    # Regular expression to find patterns like 'IN0', 'IN1', etc.
    pattern = r'IN(\d+)'

    # Replace all occurrences of the pattern in the input string
    result = re.sub(pattern, replacer, input_string)

    return result

def median_abs_deviation(arr, axis=0):
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Calculate the median along axis 0
    median = np.median(arr, axis=0)

    # Calculate the absolute deviations from the median along axis 0
    abs_deviations = np.abs(arr - median)

    # Calculate the median of the absolute deviations along axis 0
    mad = np.median(abs_deviations, axis=0)

    return mad

def count_zeros_except_first_row(array):
    # Exclude the first row
    rows_to_count = array[1:]

    # Count the number of zeros
    zero_count = np.count_nonzero(rows_to_count == 0)

    return zero_count

def count_zeros(array):
    # Count the number of zeros
    zero_count = np.count_nonzero(array == 0)

    return zero_count

def shuffle_rows(arr):
    """
    It receives an array n x m, shuffles the rows, and returns the new array.
    """
    np.random.shuffle(arr)
    return arr

def shuffle_rows_except_first(arr):
    """
    It receives an array n x m, shuffles the rows, except the first, and 
    returns the new array.
    """
    arr_copy = np.copy(arr)
    first_row = arr_copy[0]
    np.random.shuffle(arr_copy[1:])
    return np.vstack((first_row, arr_copy[1:]))

def remove_row(arr, index):
    """
    It receives an array n x m, removes the row according to the index, and 
    returns the new array.
    """
    return np.delete(arr, index, axis=0)

def add_index_column(arr):
    """
    It receives an array n x m, adds a column with the indexes, and returns 
    the new array.
    """
    row_indices = np.arange(arr.shape[0]).reshape(-1, 1)
    return np.hstack((row_indices, arr))

def remove_columns(arr, x):
    """
    It receives an array n x m and a value x. It checks the second row of the array,
    and remove the columns where the respective value is greater than x.
    """
    row = arr[1, :]
    mask = row <= x
    return arr[:, mask]

def remove_columns_with_different_value(A, x):
    """
    It receives an array n x m and a value x. It checks the second row of the array,
    and remove the columns where the respective value is different than x.
    """
    second_row = A[1]  # Extract the second row
    mask = second_row == x  # Create a boolean mask
    result = A[:, mask]  # Apply the mask to the original array
    return result

def represent_matrix_behaviour(A, threshold):
    """
    It receives a numpy array A n x m and an array threshold with length n. 
    It checks each of the columns for each row of A, and replaces the value 
    by 0 if it is smaller than the respective value in threshold.
    """
    mask = A < threshold[:, None]  # Create a boolean mask using broadcasting
    result = np.where(mask, 0, 1)  # Use np.where to replace values
    
    return result

def remove_equal_rows(A):
    """
    It receives an array A, and remove equal rows.
    """
    unique_rows, indices = np.unique(A, axis=0, return_index=True)
    result = A[indices]
    return result

def remove_equal_columns(A):
    """
    It receives an array A, and remove equal columns.
    It does not consider the first row while checking if two columns are equal 
    to each other.
    """
    transposed_A = A.T  # Transpose the array
    unique_cols, indices = np.unique(transposed_A[:,1:], axis=0, return_index=True)  # Remove first row
    result = transposed_A[indices]
    return result.T  # Transpose back to original shape

def find_equal_columns(A, column_index):
    """
    It receives an array A, and an index, and check which columns are equal
    to the column with that index.
    It ignores the first row when checking if a column is equal to another one.
    """
    equal_column_indices = np.where(np.all(A[1:] == A[1:, column_index][:, None], axis=0))[0]
    return equal_column_indices

def aggregate_rows(arr, batch_size):
    """
    It receives an array n x m, and a batch_size.
    Row 0 is kept untouched, since it contains the indexes. 
    From row 1, it aggregates the values of the rows with their average 
    according to the batch_size.
    """
    l, m = arr.shape
    n = l - 1 #first row has indexes, so we don't count it
    new_n = math.ceil(n / batch_size) + 1
    
    result = np.zeros((new_n, m))
    result[0] = arr[0]
    
    for i in range(1, new_n):
        start = (i - 1) * batch_size + 1
        end = min(i * batch_size, n)
        result[i] = np.sum(arr[start:end+1], axis=0) / (end - start + 1)
    
    return result

def aggregate_rows_sum(arr, batch_size):
    """
    It receives an array n x m, and a batch_size.
    Row 0 is kept untouched, since it contains the indexes. 
    From row 1, it aggregates the values of the rows with their addition 
    according to the batch_size.
    """
    l, m = arr.shape
    n = l - 1 #first row has indexes, so we don't count it
    new_n = math.ceil(n / batch_size) + 1
    
    result = np.zeros((new_n, m))
    result[0] = arr[0]
    
    for i in range(1, new_n):
        start = (i - 1) * batch_size + 1
        end = min(i * batch_size, n)
        result[i] = np.sum(arr[start:end+1], axis=0)
    
    return result
            
def WA(a, b, x):
    x = float(x)
    return x*a+(1-x)*b

def OWA(a, b, x):
    x = float(x)
    return x*np.maximum(a, b)+(1-x)*np.minimum(a, b)

def minimum(a, b):
    return np.minimum(a, b)

def maximum(a, b):
    return np.maximum(a, b)
    
def dilator(b):
    return b**0.5

def dilator3(b):
    return b**(1/3)

def dilator4(b):
    return b**0.25

def concentrator(b):
    return b**2

def concentrator3(b):
    return b**3

def concentrator4(b):
    return b**4

def fuzzy_AND(a, b):
    return a * b

def fuzzy_OR(a, b):
    return a + b - a*b

def complement(b):
    return 1 - b