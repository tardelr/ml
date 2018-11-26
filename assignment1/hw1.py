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
	    return np.linalg.inv((self.lmbda * np.eye(self.d)) + (self.sigma2 ** -2) * np.dot(self.X, self.X.T))

	def mu_map(self):
	    return np.dot(
				np.linalg.inv(self.lmbda * self.sigma2 * np.eye(self.d) + np.dot(self.X,self.X.T)),
				np.dot(self.X, self.y))

	def vmv_product(vector, matrix):
		return np.dot( np.dot(vector.T, matrix), vector)

	def sort_new_data(self):
	    return np.argsort(map(lambda x: vmv_product(x, self.cmatrix), self.X_test.T))

	def get_next_value(self):
		return sort_new_data()[-1]

	def active_learning(self):
		index = get_next_value
		x0 = X_test[:, index]
		y0 = y_test[index]
		self.X_train = np.column_stack([self.X_train, self.x0])
		self.y_train = np.hstack([self.y_train, self.y0])

	def part2():
	    pass

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output \
