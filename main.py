import numpy as np
import scipy.stats as sps
from datetime import datetime


def householder_matrix(x: np.ndarray):
    """
    Computes a corresponding Householder matrix by vector x.
    """
    alpha = np.linalg.norm(x)
    v = (x + alpha * np.array([1] + [0] * (x.size - 1))).reshape(x.size, 1)

    return np.identity(v.shape[0]) - (2 * np.dot(v, v.T) / np.dot(v.T, v))


def pad_before(matrix, m):
    """
        Pads a nxn square matrix with m-n ones on the main diagonal (and zeros in other positions)
        at the upper-left corner.
    """
    n, _ = matrix.shape

    left = np.vstack([np.identity(m - n), np.zeros((n, m - n))])
    right = np.vstack([np.zeros((m - n, n)), matrix])

    return np.hstack([left, right])


def qr_decompose(A):
    """
    Calculates matrices Q, R so that QR = A and R is upper-triangular.
    """
    m, n = A.shape
    assert m >= n, "In an input matrix of shape (m, n) m must not be lower than n"
    Q = np.identity(m)
    R = A
    for i in range(n):
        h_matrix = pad_before(householder_matrix(R[i:, i]), m)
        Q = np.dot(h_matrix, Q)
        R = np.dot(h_matrix, R)
    return Q.T, R


def uptriang_reverse_substitution(r, b):
    """
    Solves a least squares problem with coefficient matrix r and answer matrix b using reverse substitution.
    Returns a vector x minimizing the L2 norm of Rx-b.
    """
    m, n = r.shape

    if n == 1:
        return np.array([b[0] / r[0, 0]])
    else:
        x_n = b[n - 1] / r[n - 1, n - 1]
        return np.concatenate((uptriang_reverse_substitution(r[:-1, :-1], b[:-1] - x_n * r[:-1, -1]), [x_n]))


def solve_least_squares(A, b):
    b = b.reshape((b.size,))
    assert A.shape[0] == b.size
    q, r = qr_decompose(A)
    return uptriang_reverse_substitution(r, np.dot(q.T, b))


def main():
    num_tests = int(input("Enter the desired number of tests: "))

    print('Testing...\n')
    for i in range(num_tests):
        print('{}: '.format(i + 1))
        m = np.random.randint(low=3, high=500)
        n = np.random.randint(low=3, high=m)

        A = sps.cauchy.rvs(size=(m, n))

        print('----Generated a {} x {} matrix A with independent Cauchy variables'.format(m, n))

        b = sps.cauchy.rvs(size=m)

        print('----Generated a {} x 1 answer vector b with independent Cauchy variables'.format(m))

        start = datetime.now()
        Q, R = qr_decompose(A)
        end = datetime.now()
        time_elapsed = end - start

        discrepancy = np.linalg.norm(np.dot(Q, R) - A)

        print('----Calculated the QR decomposition of A. '
              'L2 discrepancy norm: {:0.3e}. '
              'Time elapsed: {}'.format(discrepancy, time_elapsed))

        start = datetime.now()
        x = solve_least_squares(A, b)
        end = datetime.now()
        time_elapsed = end - start

        analytical_x = np.dot(np.linalg.pinv(A), b)

        discrepancy = np.linalg.norm(x - analytical_x)

        print('----Solved a least squares problem Ax = b. '
              'L2 discrepancy norm with the solution found using pseudoinverse operator: {:0.3e}. '
              'Time elapsed: {}'.format(discrepancy, time_elapsed))


if __name__ == '__main__':
    main()
