import numpy as np
from scipy.integrate import quad
from scipy import optimize

# page internet : http://ofp.cosmo-ufes.org/

c = 299_792.458 # km/s
H0 = 68 #(km/s) / Mpc

mu, z = np.loadtxt('data.txt').T
cov = np.loadtxt("jla_mub_covmatrix.txt").reshape((len(mu), len(mu)))

def hubble(zi, o_m, ol, w):
    return 1 / np.sqrt(o_m * (1 + zi)**3 + ol * (1 + zi)**(3 * (1 + w)))

def dl(o_m, ol, w=1):
    chi = np.array([quad(hubble, 0, zmax, args=(o_m, ol, w))[0] for zmax in z])
    return (1 + z) * (c / H0) * chi

def residuals(om, ol, w):
    return mu - (5 * np.log10(dl(om, ol, w)) + 25)

# cov_inv = np.linalg.inv(cov)
def chi2(params):
    om, ol, w = params
    residus = mu - (5 * np.log10(dl(om, ol, w)) + 25)
    return residus.T @ cov @ residus


# def densite_totale(params):
#     return 1 - (params[0] + params[1])

#  constraints = ({'type':'eq', 'fun':densite_totale}, ),

if __name__ == "__main__":

    bounds = ((0, 1.5), (0, 1.5), (-2, 2))
    guess = np.array([0.3, 0.7, -1])
    best_params = optimize.minimize(chi2, guess, bounds=bounds, method = 'SLSQP', tol=10**-10)
    om, ol, w = best_params.x
    # om = 0.3
    # ol = 0.69
    # w=-1
    q = 0.5 * om + 0.5 * (1 + 3 * w) * ol
    print(f"For {H0=}")
    print(f"Best fit for \omega_m = {om:.2f}")
    print(f"Best fit for \omega_l = {ol: .2f}")
    print(f"Best fit for w = {w : .2f}")
    print(f"This yields {q=:.2f}")