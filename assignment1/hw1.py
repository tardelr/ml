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
def trick_to_avoid_sum(matrix):
	"""
	Avoid sum of inner products like Sum(xi * xi.T)
	"""
	d,_ = martix.shape
	return sum(matrix * matrix.T * np.eye(d))

def covariance_matrix(X,y,lmbda,sig):
	n,d = X.shape
	return np.linalg.inv((lmbda * np.eye(d)) + (sig ** -2) * X * X.T)

def mu_map(lmbda, sigma2, X, y):
	n,d = X.shape
	return np.linalg.inv(lmbda * sigma2 * np.eye(d) + X.T*X) * np.dot(X.T, y)

def covariance_posterior(X, x0, lmbda, sigma2_input):
	n,d = X.shape
	inner_part= (x0 * x0.T + trick_to_avoid_sum(X))
	return 	np.linalg.inv(lmbda * np.eye(d) + (sigma2_input ** -1) * inner_part)

def mu_posterior(X, y, x0, y0, lmbda, sigma2_input):
	inner_part_first = (x0 * x0.T + trick_to_avoid_sum(X))
	first_term = np.linalg.inv(lmbda * sigma2_input * np.eye(d) + inner_part_first)
	second_term = np.dot(x0, y0) + sum(X * y.T * np.eye(d))
	return 	first_term + second_term

def part2():
    pass

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file




