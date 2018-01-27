import sys
import copy
import numpy as np
import pickle


def rechlon(A):
    A = copy.deepcopy(A)
    m, n = A.shape
    for k in range(min(m, n - 1)):
        i_max = k + np.argmax(np.abs(A[k:, k]))
        if A[i_max, k] == 0:
            continue

        A[[i_max, k]] = A[[k, i_max]]

        for i in range(k + 1, m):
            f = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] = A[i, j] - A[k, j] * f

    return A


def backsubstitution(T):
    m, n = T.shape
    if n != m + 1:
        raise ValueError("Back substitution must get array of shape : m, m +1")

    x = np.zeros(m)
    for i in reversed(range(m)):
        k = np.dot(x, T[i,:-1])
        x[i] = (T[i, n - 1] - k) / T[i, i]

    return x


def gauss_elimination(Ab):
    Ab = copy.deepcopy(Ab)
    Ab = Ab.astype(np.float64)

    R = rechlon(Ab)
    m, n = R.shape
    x_len = n - 1
    for free_i in reversed(range(m)):
        if R[free_i, n - 1 - 1] != 0:
            break

        if R[free_i, n - 1] != 0:
            raise ValueError("Inconsistent system of equations: no solutions exist.")
    T = R[:free_i + 1, :]
    S = T[:, :free_i + 1]
    b_row = T[:, n - 1]
    b_row1 = b_row.reshape((len(b_row), 1))
    S = np.hstack((S, b_row1))
    x_unique = backsubstitution(S)
    x = np.pad(x_unique, (0, x_len - len(x_unique)), 'constant')
    return x


if __name__ == "__main__":
    args = sys.argv[1]
    Ab = pickle.load(open(args))
    x = gauss_elimination(Ab)
    print("The solution is: %s" % x)
    
    output_file = "./output.pickle"
    print("Writing the solution array to %s" % output_file)
    pickle.dump(x, open(output_file, "w"))
