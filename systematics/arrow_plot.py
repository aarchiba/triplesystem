import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from collections import namedtuple
from pylab import rcParams
rcParams['savefig.dpi'] = 144
import pickle
import math
from scipy.stats import norm
import matplotlib.mlab as mlab
from scipy import stats
from matplotlib.font_manager import FontProperties

def arrow_plot(par_dict, mjd, phase, unc):
    """ par_dict is dictionary with Best_parameters.
mjd is mjd, phase is residuals and unc is its uncertainty (units is phase of the pulsar!)
This function gives you the arrow plot and the VALUE of the biggest arrow in MICRO SECONDS
value=arrow_plot(par_dict, mjd, phase, unc)"""
    #-----function-1 --- make a matrix
    ############################################
    #input:
    #ll - nhumber of inner orbit turns
    #mjd - input array...

    matrix_out=namedtuple("matrix",
           ["names","A","Adict"])

    def matrix(par_dict, ll, mjd):
        Adict = {}
        pb_i = par_dict['pb_i']
        pb_o = par_dict['pb_o']
        phi_i=(2.0*math.pi)/pb_i
        phi_o=(2.0*math.pi)/pb_o
        for i in range(ll):
            for j in range(-(ll-1),ll):
                if i==0 and j<0:
                    continue
                Adict[i,j,'cos'] = np.cos((i*phi_o+j*phi_i)*mjd)
                if (i,j)!=(0,0):
                    Adict[i,j,'sin'] = np.sin((i*phi_o+j*phi_i)*mjd)
        names = sorted(Adict.keys())
        A = np.array([Adict[n] for n in names]).T
        return matrix_out(names=names, A=A, Adict=Adict)

    #-----function-2 --- fit A using matrix function
    ###################################################

    def linear_least_squares_cov(par_dict, Acols, mjd, phase, unc):
        A = matrix(par_dict, Acols, mjd)
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

    #function-3---create_plotable_arrows
    ####################################
    plotter_out = namedtuple("plotter_out",
        ["X","Y","U","V","M","err_X","err_Y","err_ix"])

    def plotter(par_dict, Acols, mjd, phase, unc):
        ang = np.linspace(0,2*np.pi,100)
        circle = np.array([np.cos(ang),np.sin(ang)])
        fit = linear_least_squares_cov(par_dict, Acols, mjd, phase, unc)
        result = dict(zip(fit.names,fit.x))
        X = []
        Y = []
        U = []
        V = []
        err_X = []
        err_Y = []
        err_ix = []
        for (i,j,f) in result.keys():
            Y.append(i)
            X.append(j)
            U.append(result[i,j,'cos'])
            err_ix.append((i,j))
            if (i,j)==(0,0):
                V.append(0)
                err_X.append(0)
                err_Y.append(0)
            else:
                V.append(result[i,j,'sin'])

                st=fit.names.index((i, j, 'cos'))
                fin=fit.names.index((i, j, 'sin'))+1
                err_M=fit.cov[st:fin,st:fin]
                assert err_M.shape==(2,2)
                L = scipy.linalg.cholesky(err_M)
                Lcircle = np.dot(L,circle)
                # print Lcircle[0][7]
                err_X.append(Lcircle[0])
                err_Y.append(Lcircle[1])

        M = np.hypot(V, U)
        return plotter_out(X=X, Y=Y, U=U, V=V, M=M, err_X=err_X, err_Y=err_Y, err_ix=err_ix)    

    #function-4---make-a-plot
    ####################################

    font0 = FontProperties()
    font0.set_size('10')
    font0.set_weight('normal')
    font0.set_family('serif')
    font0.set_style('normal')

    err_red=np.array([191, 54, 12])/255.0
    err_bl=np.array([21, 101, 192])/255.0

    def draw_nocb_plot(par_dict, Acols, mjd, phase, unc, scale=None):
        pl=plotter(par_dict, Acols, mjd, phase, unc)
        if scale is None:
            ar_scale=np.amax(pl.M)
        else:
            ar_scale=scale
        for k in range(0,len(pl.err_X)):
            i,j = pl.err_ix[k]
            plt.plot(pl.err_X[k]/ar_scale+j,pl.err_Y[k]/ar_scale+i, color=err_bl, lw=1, alpha=0.3)
        Q = plt.quiver(pl.X, pl.Y, pl.U, pl.V, units='x', scale=ar_scale, color=err_bl)
        plt.xlim(-Acols+0.05,Acols-0.05)
        plt.ylim(-0.95,Acols-0.05)
        plt.xlabel('Inner orbit frequencies')
        plt.ylabel('Outer orbit frequencies')
        plt.gca().set_aspect('equal')
        err_red=np.array([191, 54, 12])/255.0
        plt.figtext(0.15,0.2, r'the longest arrow = %4.3f $\mu$s'%(ar_scale), fontproperties=font0, color=err_red)
        return ar_scale
    

    
    p_period=0.00273258863228
    scl=p_period*1e6
    value=draw_nocb_plot(par_dict, 4, mjd, phase*scl, unc*scl)
    
    return value
