import numpy as np
import scipy.optimize

def neldermead(func, x0s,
               ftol=1e-2, maxfev=500):
    """
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm. Based on the code from scipy with tuned
    termination criteria.
    """

    fcalls = 0
    x0s = np.asarray(x0s)
    M, N = x0s.shape
    if M!=N+1:
        raise ValueError("x0s must be N+1 points of dimension N")

    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    one2np1 = list(range(1, N + 1))

    sim = np.zeros((N + 1, N), dtype=x0s.dtype)
    fsim = np.zeros((N + 1,), float)
    for i in range(N+1):
        sim[i] = x0s[i]
        fsim[i] = func(sim[i])
        fcalls += 1

    # sort so sim[0,:] has the lowest function value
    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind, 0)
    sim = np.take(sim, ind, 0)

    while (fcalls < maxfev):
        if np.max(np.abs(fsim[0] - fsim[1:])) <= ftol:
            break

        xbar = np.add.reduce(sim[:-1], 0) / N
        xr = (1 + rho)*xbar - rho*sim[-1]
        fxr = func(xr)
        fcalls += 1
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1 + rho*chi)*xbar - rho*chi*sim[-1]
            fxe = func(xe)
            fcalls += 1

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + psi*rho)*xbar - psi*rho*sim[-1]
                    fxc = func(xc)
                    fcalls += 1

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi)*xbar + psi*sim[-1]
                    fxcc = func(xcc)
                    fcalls += 1

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma*(sim[j] - sim[0])
                        fsim[j] = func(sim[j])
                        fcalls += 1

        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)

    x = sim[0]
    fval = fsim[0]
    return x, fval
