import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

## Solution for Part 1
def ridge_regression(X_train, y_train, lambda_input):
	n,d = X_train.shape
	inner_part = np.linalg.inv(lambda_input*np.eye(n) + X_train*X_train.T)
	return np.dot(inner_part * X_train, y_train)

wRR = ridge_regression(X_train, y_train, input_lambda)
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
class activeLearning:

	def __init__(X, y, X0, y0):
		aw

	def sort_new_data(new_data):
	    return

	def vmv_product(vector, matrix):
	    return np.dot( np.dot(vector.T, matrix), vector)

class ridgeRegression:

	def __init__(self, X_train, y_train, X_test, y_test, sigma2, lmbda):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.sigma2 = sigma2
		self.lmbda = lmbda
		self.n, self.d = X_train.shape
		self.cmatrix = None
		self.mu = None
		self.wRR = None

	def covariance_matrix(self):
	    return np.linalg.inv((lmbda * np.eye(d)) + (sig ** -2) * np.dot(X, X.T))

	def mu_map(lmbda, sigma2, X, y):
	    d,n = X.shape
	    return np.dot(np.linalg.inv(lmbda * sigma2 * np.eye(d) + np.dot(X,X.T)), np.dot(X, y))

	def part2():
	    pass

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output \
