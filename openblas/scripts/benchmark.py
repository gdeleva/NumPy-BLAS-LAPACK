"""
Program to measure the time that NumPy takes
to perform common linear algebra routines
supported by BLAS and LAPACK libraries
"""

import time
import numpy as np
from numpy.linalg import inv
from numpy.linalg import qr
from numpy.linalg import svd
from numpy.linalg import cholesky
from numpy.linalg import eig

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_spd_matrix

np.show_config()

def dot_product_timer(x_shape=(5000, 5000),
                      y_shape=(5000, 5000),
                      mean=0,
                      std=10,
                      seed=8053):
  """
  A timer for the formula array1.dot(array2).

  Inputs:
    x_shape: Tuple of 2 Int
             Shape of array1;
    y_shape: Tuple of 2 Int
             Shape of array2;
    mean: Float
          Mean of the normal distribution used for random
          selection of elements of array1 and array2;
    std: Float
         Standard deviation of the normal distribution used for random
         selection of elements of array1 and array2;
    seed: Int
          Seed used in np.random.seed

  Outputs:
    delta_t: Float
             Number of seconds for completing array1.dot(array2);
  """
  np.random.seed(seed)
  array1 = np.random.normal(mean, std, x_shape)
  array2 = np.random.normal(mean, std, y_shape)
  start = time.time()
  array1.dot(array2)
  delta_t = time.time() - start

  return delta_t

def sym_matrix_eig_timer(row_n=2000,
                         mean=0,
                         std=10,
                         seed=8053):
  """
  A timer for calculating the eigenvalues of a symmetric matrix.
  The symmetric matrix is defined as (matrix + matrix.T)/2,
  where the elements of matrix are randomly selected.

  Inputs:
    row_n: Int
           Number of rows and columns of matrix;
    mean: Float
          Mean of the normal distribution used for random
          selection of elements of matrix;
    std: Float
         Standard deviation of the normal distribution used for random
         selection of elements of matrix;
    seed: Int
          Seed used in np.random.seed

  Outputs:
    delta_t: Float
             Number of seconds for completing eig(sym_matrix);
  """
  np.random.seed(seed)
  matrix = np.random.normal(mean, std, (row_n, row_n))
  sym_matrix = (matrix + matrix.T)/2
  start = time.time()
  eig(sym_matrix)
  delta_t = time.time() - start

  return delta_t

def sym_matrix_cholesky_timer(row_n=2000,
                              seed=8053):
  """
  A timer for calculating the Cholesky decomposition of a symmetric matrix.
  The symmetric matrix is generated using sklearn.datasets.make_spd_matrix.

  Inputs:
    row_n: Int
           Number of rows and columns of matrix;
    seed: Int
          Seed used in sklearn.datasets.make_spd_matrix

  Outputs:
    delta_t: Float
             Number of seconds for completing cholesky(sym_matrix);
  """
  np.random.seed(seed)
  sym_matrix = make_spd_matrix(row_n, random_state=seed)
  start = time.time()
  cholesky(sym_matrix)
  delta_t = time.time() - start

  return delta_t

def svd_timer(shape=(2000, 1000),
              mean=0,
              std=10,
              seed=8053):
  """
  A timer for calculating the singular value decomposition of a matrix whose elements
  are randomly selected.

  Inputs:
    shape: Tuple of 2 Int
           Shape of the matrix;
    mean: Float
          Mean of the normal distribution used for random
          selection of elements of the matrix;
    std: Float
         Standard deviation of the normal distribution used for random
         selection of elements of the matrix;
    seed: Int
          Seed used in np.random.seed

  Outputs:
    delta_t: Float
             Number of seconds for completing svd(matrix);
  """
  np.random.seed(seed)
  matrix = np.random.normal(mean, std, shape)
  start = time.time()
  svd(matrix)
  delta_t = time.time() - start

  return delta_t

def qr_timer(shape=(2000, 1000),
             mean=0,
             std=10,
             seed=8053):
  """
  A timer for calculating the qr factorization of a matrix whose elements
  are randomly selected.

  Inputs:
    shape: Tuple of 2 Int
           Shape of the matrix;
    mean: Float
          Mean of the normal distribution used for random
          selection of elements of the matrix;
    std: Float
         Standard deviation of the normal distribution used for random
         selection of elements of the matrix;
    seed: Int
          Seed used in np.random.seed

  Outputs:
    delta_t: Float
             Number of seconds for completing qr(matrix);
  """
  np.random.seed(seed)
  matrix = np.random.normal(mean, std, shape)
  start = time.time()
  qr(matrix)
  delta_t = time.time() - start

  return delta_t

def linear_regression_timer(obs_n=2000000,
                            var_n=150,
                            coeff_mean=0,
                            coeff_std=10,
                            x_std=20,
                            error_std=1,
                            check_result=False,
                            seed=8053):
  """
  A timer of the formula inv(matrix_x.T.dot(matrix_x)).dot(matrix_x.T).dot(vector_y)
  used to calculate the coefficients of a linear regression model.

  NOTE: We only use this formula for testing numpy speed.
  This formula is not recommended for parameter estimation.
  We use the recommended sklearn.linear_model.LinearRegression
  for testing the result

  Inputs:
    obs_n: Int
           Number of rows of matrix_x;
    var_n: Int
           Number of columns of matrix_x;
    coeff_mean: Float
                Mean of the normal distribution used for random
                selection of var_n coefficients;
    coeff_std: Float
              Standard deviation of the normal distribution used for random
              selection of var_n coefficients;
    x_std: Float
           standard deviation of the normal distribution used for random
           selection of the entries of matrix_x. The mean of this distribution is zero.
    error_std: Float
               standard deviation of the residuals;
    check_result: Boolean
                  If True sklearn.linear_model.LinearRegression
                  is used to check the accuracy of
                  inv(matrix_x.T.dot(matrix_x)).dot(matrix_x.T).dot(vector_y).
    seed: Int
          Seed used in np.random.seed

  Outputs:
    delta_t: Float
             Number of seconds for completing the calculation of
             inv(matrix_x.T.dot(matrix_x)).dot(matrix_x.T).dot(vector_y);
  """

  np.random.seed(seed)
  coefficients = np.random.normal(coeff_mean, coeff_std, var_n)
  errors = np.random.normal(0, error_std, obs_n)
  matrix_x = np.random.normal(0, x_std, (obs_n, var_n))
  vector_y = matrix_x.dot(coefficients)+errors

  start = time.time()
  coef1 = inv(matrix_x.T.dot(matrix_x)).dot(matrix_x.T).dot(vector_y)
  delta_t = time.time() - start

  if check_result:
    reg = LinearRegression(fit_intercept=False).fit(matrix_x, vector_y)
    max_error = np.max(np.abs(coef1-reg.coef_))
    print(max_error)

  return delta_t

def my_test(function_list,
            repeat=10,
            percentiles=(0, 5, 10, 50, 90, 95, 100)):
  """
  A function that calls other test functions and prints time percentiles for
  each test function,

  Inputs:
    function_list: List of functions
                   List of test functions
    n: Number of tests. This is the length of the vector that contains time measurements
      for each test function;
    percentiles: Tuple of floats
                List of percentiles to compute on the array that contains n time measurements
                for each test function

  Outputs:
    time_percentiles: Array of floats with shape (len(function_list), len(percentiles))
                      Each entry i,j is the q-th percentile of time measurements for the
                      i-th function of function_list where q is the j-th value of percentiles.
  """

  func_num = len(function_list)
  time_measurements = np.zeros((func_num, repeat))
  time_percentiles = np.zeros((func_num, len(percentiles)))

  for count in range(repeat):
    for func_count in range(func_num):
      time_measurements[func_count, count] = function_list[func_count]()


  for func_count in range(func_num):
    time_percentiles[func_count] = np.percentile(time_measurements[func_count], percentiles)

  return time_percentiles


if __name__ == "__main__":
  def scalar_product():
    """
    A wrapper for dot_product_timer in order to define a scalar product between arrays.
    """
    x_shape=(1000000,)
    delta_t = dot_product_timer(x_shape=x_shape, y_shape=x_shape)
    return delta_t

  function_lst = [scalar_product, dot_product_timer, linear_regression_timer,
                  svd_timer, qr_timer, sym_matrix_cholesky_timer, sym_matrix_eig_timer]
  t_percentiles = my_test(function_list=function_lst)

  # Print results
  L = len(function_lst)
  for f_count in range(L):
    print(function_lst[f_count].__name__,
          " ".join(["{:.2f}".format(x) for x in t_percentiles[f_count]]))
