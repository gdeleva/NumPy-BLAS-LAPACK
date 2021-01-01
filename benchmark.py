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


np.show_config()

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
    std_mean: Float
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
            repeat=5,
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
  function_lst = [linear_regression_timer]
  t_percentiles = my_test(function_list=function_lst)

  # Print results
  L = len(function_lst)
  for f_count in range(L):
    print(function_lst[f_count].__name__,
          " ".join(["{:.2f}".format(x) for x in t_percentiles[f_count]]))
