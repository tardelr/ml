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
	inner_part = np.linalg.inv(lambda_input*np.eye(d) + X_train*X_train.T)
	return np.dot(inner_part * X_train, y_train)

wRR = ridge_regression(X_train, y_train, lambda_input)
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def covariance_matrix(X,y,lmbda,sig):
	n,d = X.shape
	return np.linalg.inv((lmbda * np.eye(d)) + (sig ** -2) * X * X.T)

def sum_of_outer_products(X):
	outer_prod_list = []
	for i in range(X.shape[0]):
		outer_prod_list.append(np.outer(X[i], X[i].T))
	return sum(outer_prod_list)

def posterior_covariance_matrix(X,lmbda,sig,x0):
	_,d = X.shape
	return np.linalg.inv(lmbda*np.eye(d) + (sig ** -2) * (np.outer(x0,x0.T) + sum_of_outer_products(X)))

def part2():
    cmatrix = covariance_matrix(X_train, y_train, lambda_input, sigma2_input)
    index0 = []
    sig_list = []
    sigma2 = sigma2_input
    while len(index0) < 10:
		for i in range(X0.shape[0]):
			x0 = X0[i]
			post_covariance = posterior_covariance_matrix(X, lambda_input, sigma2_input,x0)
			sig_list.append(sigma2 + np.inner(np.inner(x0, post_covariance), x0))
		idx = np.argmax(sig_list)
		X = np.vstack([X, X0[idx]])
		X0[idx] = False
		index0.append(idx)
		sigma2 = sig_list[idx]
    return index0

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file
