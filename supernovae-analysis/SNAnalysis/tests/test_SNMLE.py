import unittest
import numpy as np
from scipy import linalg
from numpy.testing import assert_array_almost_equal
from essais.SNMLE_analysis import sigma_d, sigma_l, A, residuals, magnitude_distance, log_likelihood
from essais.params import N, pre_found_best, path_covmat, path_Z


def COV(alpha, beta, sig_M, sig_x, sig_c):  # Total covariance matrix
    block3 = np.array([[sig_M + sig_x * alpha ** 2 + sig_c * beta ** 2, -sig_x * alpha, sig_c * beta],
                       [-sig_x * alpha, sig_x, 0],
                       [sig_c * beta, 0, sig_c]])
    return linalg.block_diag(*[block3 for _ in range(740)])


def JLA_residual(Z_data, parameters):  # Total residual, \hat Z - Y_0*A
    omega_m, omega_l, alpha, beta, M0, X0, C0 = parameters[[0, 1, 2, 5, 8, 3, 6]]
    Y0A = np.array([M0 - alpha * X0 + beta * C0, X0, C0])
    mu = magnitude_distance(omega_m, omega_l)[0]
    return np.hstack([(Z_data[j, 1:4] - np.array([mu[j], 0, 0]) - Y0A) for j in range(N)])


def m2loglike(pars):
    cov = COV(*[pars[j] for j in [2, 5, 9, 4, 7]]) + sigma_d(path_covmat)
    try:
        chol_fac = linalg.cho_factor(cov, overwrite_a=True, lower=True)
    except np.linalg.linalg.LinAlgError:  # If not positive definite
        return 13993 * 10. ** 20
    except ValueError:  # If contains infinity
        return 13995 * 10. ** 20
    res = residuals(pre_found_best)

    # Don't throw away the logPI part.
    part_log = 3 * N * np.log(2 * np.pi) + np.sum(np.log(np.diag(chol_fac[0]))) * 2
    part_exp = np.dot(res, linalg.cho_solve(chol_fac, res))

    return part_log  + part_exp


class MyTestCase(unittest.TestCase):

    @staticmethod
    def test_sigma_d():
        COVd = np.load(path_covmat + "stat.npy")
        for i in ["cal", "model", "bias", "dust", "pecvel", "sigmaz", "sigmalens", "nonia"]:
            COVd += np.load(path_covmat + i + '.npy')

        assert_array_almost_equal(sigma_d(path=path_covmat), COVd)

    def test_shape_sigma_d(self):
        matrice = sigma_d(path=path_covmat)
        self.assertEqual(matrice.shape, (3 * N, 3 * N))

    @staticmethod
    def test_sigma_l():
        test_array = np.diag(np.array([1.17007078e-02, 8.67848219e-01, 5.04364259e-03] * N))
        assert_array_almost_equal(sigma_l(pre_found_best), test_array)

    def test_shape_sigma_l(self):
        matrice = sigma_l(pre_found_best)
        self.assertEqual(matrice.shape, (3 * N, 3 * N))

    @staticmethod
    def test_At_sigma_l_A():
        """This is the product (At*Sigma_d*A) we often find. And the only way to test my code
            against the one from Nielsen et al.
        """
        ATsiglA = COV(1.34469382e-01, 3.05861386e+00, 1.17007078e-02, 8.67848219e-01, 5.04364259e-03)
        a = A(pre_found_best)
        sig_l = sigma_l(pre_found_best)
        built_AtsiglA = np.dot(a.T, np.dot(sig_l, a))
        assert_array_almost_equal(ATsiglA, built_AtsiglA)

    @staticmethod
    def test_residuals():
        Z_data = np.load(path_Z)
        residuals_nielsen = JLA_residual(Z_data, pre_found_best)
        my_residuals = residuals(pre_found_best)
        assert_array_almost_equal(my_residuals, residuals_nielsen)

    @staticmethod
    def test_total_covariance_matrix():
        a = A(pre_found_best)
        sig_l = sigma_l(pre_found_best)
        sig_d = sigma_d(path_covmat)   # We know that this function produces the same result as in Nielsen (Covd)
        ATsiglA = COV(1.34469382e-01, 3.05861386e+00, 1.17007078e-02, 8.67848219e-01, 5.04364259e-03)
        total_cov_matrix = sig_d + np.dot(a.T, np.dot(sig_l, a))
        assert_array_almost_equal(ATsiglA + sig_d, total_cov_matrix)

    def test_log_likelihood(self):
        log_L = log_likelihood(pre_found_best)
        nielsen_logL = m2loglike(pre_found_best)
        self.assertEqual(nielsen_logL, log_L)


if __name__ == '__main__':
    unittest.main()
