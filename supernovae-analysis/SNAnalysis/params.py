import numpy as np

c = 299792.458  # km/s
H0 = 70  # (km/s) / Mpc
N = 740  # Number of SNe

# parameters = {o_m, o_l, a, x0, var_x0, b, c0, var_c0, M0, var_M0}
pre_found_best = np.array(
            [3.40658319e-01, 5.68558786e-01, 1.34469382e-01, 3.84466029e-02, 8.67848219e-01, 3.05861386e+00,
             -1.59939791e-02, 5.04364259e-03, -1.90515806e+01, 1.17007078e-02])
path_covmat = "/home/lucas/Documents/Master/M1/T-D Astrophysique-20220930T141717Z-001/T-D Astrophysique/SNMLE/covmat/"
path_Z = "/home/lucas/Documents/Master/M1/T-D Astrophysique-20220930T141717Z-001/T-D Astrophysique/SNMLE/JLA.npy"
path_interpolation = "/home/lucas/Documents/Master/M1/T-D Astrophysique-20220930T141717Z-001/T-D Astrophysique/SNMLE/Interpolation.npy"
