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
def covariance_matrix(X,y,lmbda,sig):
	n,d = X.shape
	return np.linalg.inv((lmbda * np.eye(d)) + (sig ** -2) * X * X.T)



def part2():
    pass

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file
