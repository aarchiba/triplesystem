import numpy as np
import scipy.linalg
import math
from scipy.stats import norm
from scipy import stats
import matrix_fun as mtrx


def linear_least_squares_cov(par_dict, Acols, mjd, phase, unc):
    A = mtrx.matrix(par_dict, Acols, mjd)
    Adict=A.Adict
    names=A.names
    x, chi2, rk, s = scipy.linalg.lstsq(A.A/unc[:,np.newaxis], phase/unc)
    res = phase-np.dot(A.A,x)
    Ascaled = A.A/unc[:,np.newaxis]

    cov = np.linalg.pinv(np.dot(Ascaled.T,Ascaled))
    n=len(phase)
    cov_scaled = cov*chi2/n

    class Result(object):
        pass
    r = Result()
    r.names = names
    r.x = x
    r.chi2 = chi2
    r.rk = rk
    r.s = s
    r.res = rescov = np.linalg.pinv(np.dot(Ascaled.T,Ascaled))
    n=len(phase)
    r.A = A
    r.Adict = Adict
    r.cov = cov
    r.cov_scaled = cov_scaled

    return r
