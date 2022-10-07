import numpy as np
from scipy import linalg, interpolate
from torch import cholesky_inverse
from params import N, c, H0, path_interpolation, path_Z, path_covmat, pre_found_best


def light_distance(omega_m, omega_l, path=path_interpolation):
    """Returns in same order as always - c/H0 multiplied on after, in mu"""
    interp = np.load(path)
    tempInt = []
    for i in range(N):
        tempInt.append(interpolate.RectBivariateSpline(np.arange(0, 1.51, .01), np.arange(-.50, 1.51, .01), interp[i]))
    return np.hstack([tempdL(omega_m, omega_l) for tempdL in tempInt])


def magnitude_distance(OM, OL):
    return 5 * np.log10(c / H0 * light_distance(OM, OL)) + 25


def get_Z_data(path=path_Z):
    return np.load(path)

# Definition de la likelihood:
# |2\pi * \Sigma_d + A.T \Sigma_l A\^(-1/2) * exp(-(Z-Y0.A)(\Sigma_d + A.T \Sigma_l A)(Z-Y0A)


def sigma_d(path=path_covmat):
    """Estimated covariance matrix, from Betoule et al."""
    covd = np.load(f'{path}/stat.npy')  # Constructing data covariance matrix w/ sys.
    # sigmaz and sigmalens are constructed as described in the JLA paper
    # all others are taken from their .tar and converted to python format
    for i in ["cal", "model", "bias", "dust", "pecvel", "sigmaz", "sigmalens", "nonia"]:
        # Notice the lack of "host" covariances - we don't include the mass-step correction.
        covd += np.load(path + i + '.npy')
    return covd


def sigma_l(parameters):
    """ Matrix containing the uncertainties on the fitted parameters.
        Arguments :
        parameters = {o_m, o_l, a, x0, var_x0, b, c0, var_c0, M0, var_M0}
        Returns: a diagonal (3x740)x(3x740)=(2220x2220) matrix
    """
    diagonal = np.array([parameters[j] for j in [9, 4, 7]] * N)
    return np.diag(diagonal)


def A(parameters):
    """Block matrix A. See the article"""
    alpha, beta = parameters[2], parameters[5]
    block = np.array([[1, 0, 0], [-alpha, 1, 0], [beta, 0, 1]])
    return linalg.block_diag(*[block for _ in range(N)])


def Y0(parameters):
    return np.array([parameters[j] for j in [8, 3, 6]] * N)


def residuals(parameters, Z_data=None):
    """Total residual, \\hat Z - Y_0*A"""
    if Z_data is None:
        Z_data = get_Z_data()
    omega_m, omega_l = parameters[0], parameters[1]
    y0 = Y0(parameters)
    a = A(parameters)
    y0A = np.dot(y0, a)[:3]  # seules les 3 premières valeurs sont différentes
    mu = magnitude_distance(omega_m, omega_l)[0]
    return np.hstack([(Z_data[j, 1:4] - np.array([mu[j], 0, 0]) - y0A) for j in range(N)])

def total_covariance(parameters):
    a = A(parameters)
    sig_d = sigma_d(path_covmat)
    sig_l = sigma_l(parameters)
    total_cov_matrix = sig_d + np.dot(a.T, np.dot(sig_l, a))
    return total_cov_matrix


def log_determinant(array):
    chol_fac = linalg.cho_factor(array, overwrite_a=True, lower=True)
    diago_matrice_triangulaire = np.diag(chol_fac[0])
    det = np.sum(np.log(diago_matrice_triangulaire)) * 2
    return det


def log_inverse(array1, array2):
    chol_factor = linalg.cho_factor(array1, overwrite_a=True, lower=True)
    return linalg.cho_solve(chol_factor, array2)


def log_likelihood(parameters, Z_data=None):
    # We first create the matrices

    a = A(parameters)
    sig_d = sigma_d(path_covmat)
    sig_l = sigma_l(parameters)
    total_cov_matrix = sig_d + np.dot(a.T, np.dot(sig_l, a))

    try:
        chol_fac = linalg.cho_factor(total_cov_matrix, overwrite_a=True, lower=True)
    except np.linalg.linalg.LinAlgError:  # If not positive definite
        return 13993 * 10. ** 20
    except ValueError:  # If contains infinity
        return 13995 * 10. ** 20

    res = residuals(parameters, Z_data)

    # factor = 3 * N * np.log(2 * np.pi) + log_determinant(total_cov_matrix)
    # exponential = np.dot(res, log_inverse(total_cov_matrix, res))
    factor = 3 * N * np.log(2 * np.pi) + np.sum(np.log(np.diag(chol_fac[0]))) * 2
    exponential = np.dot(res, linalg.cho_solve(chol_fac, res))
    # Renvoie -2 log ... D'où la disparition du 1/2 de factor et du - d'exponential
    return factor + exponential


def pulls(parameters):
    # TODO: pulls = (Z - Y0A)U^-1, complete the formula
    tot_cov = total_covariance(parameters)
    cholesky_mat = chol_fac = linalg.cho_factor(tot_cov, overwrite_a=True, lower=True)[0]
    return linalg.cho_solve(chol_fac, cholesky_mat)

def COV_C(alpha, beta, sig_M):
    cov_mat = sigma_d()
    block1 = np.array([1, alpha, -beta])
    AJLA = linalg.block_diag(*[block1 for _ in range(N)])
    return np.dot(AJLA, np.dot(cov_mat, AJLA.transpose())) + np.eye(N) * sig_M


def RES_C(omega_m, omega_l, alpha, beta, M0):
    Z = get_Z_data()
    mu = magnitude_distance(omega_m, omega_l)[0]
    return Z[:, 1] - M0 + alpha * Z[:, 2] - beta * Z[:, 3] - mu


# INPUT HERE IS REDUCED: pars = [om, ol, a, b, m0] , VM separate

def chi2_C(pars, sig_M):
    if pars[0] < 0 or pars[0] > 1.5 or pars[1] < -.50 or pars[1] > 1.5 \
            or sig_M < 0:
        return 14994 * 10. ** 20
    cov = COV_C(pars[2], pars[3], sig_M)
    cholesky_fac = linalg.cho_factor(cov, overwrite_a=True, lower=True)

    res = RES_C(*pars)

    part_exp = np.dot(res, linalg.cho_solve(cholesky_fac, res))
    return part_exp


if __name__ == "__main__":
    # print(log_likelihood(parameters=pre_found_best))
    import matplotlib.pyplot as plt
    plt.hist(pulls(pre_found_best))