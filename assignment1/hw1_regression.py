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
    inner_part = np.linalg.inv(lambda_input*np.eye(d) + np.dot(X_train.T, X_train))
    return np.dot( np.dot(inner_part, X_train.T), y_train)

wRR = ridge_regression(X_train, y_train, lambda_input)
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def covariance_matrix(X,y,lmbda,sig):
    n,d = X.shape
    return np.linalg.inv((lmbda * np.eye(d)) + (sig ** -2) * np.dot(X.T, X))

def sum_of_outer_products(X):
    outer_prod_list = []
    for i in range(X.shape[0]):
        outer_prod_list.append(np.outer(X[i], X[i].T))
    return sum(outer_prod_list)

def temp_covariance_matrix(X,lmbda,sig,x0):
    _,d = X.shape
    return np.linalg.inv(lmbda*np.eye(d) + (sig ** -2) * (np.outer(x0,x0.T) + sum_of_outer_products(X)))

def part2(X_train, y_train, lambda_input, sigma2_input, X_test):
    post_covariance = covariance_matrix(X_train, y_train, lambda_input, sigma2_input)
    index0 = []
    sigma2 = sigma2_input
    while len(index0) <= 10:
        sig_list = []
        for i in range(X_test.shape[0]):
            x0 = X_test[i]
            # post_covariance = temp_covariance_matrix(X_train, lambda_input, sigma2_input, x0)
            post_covariance = np.linalg.inv( np.linalg.inv(post_covariance) + (sigma2_input**-2)*np.dot(x0,x0.T))
            sig_list.append(sigma2 + np.inner(np.inner(x0, post_covariance), x0))
        idx = np.argmax(sig_list)
        X_train = np.vstack([X_train, X_test[idx]])
        X_test[idx] = False
        index0.append(idx)
        post_covariance = np.linalg.inv( np.linalg.inv(post_covariance) + (sigma2_input**-2)*np.dot(x0,x0.T))

    return list(np.array(index0)+1)

active = part2(X_train, y_train, lambda_input, sigma2_input, X_test)  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file
