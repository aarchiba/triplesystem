import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['savefig.dpi'] = 144
import math
import matplotlib.mlab as mlab
from matplotlib.font_manager import FontProperties
import plotter_fun as pltr

font0 = FontProperties()
font0.set_size('10')
font0.set_weight('normal')
font0.set_family('serif')
font0.set_style('normal')

err_red=np.array([191, 54, 12])/255.0
err_bl=np.array([21, 101, 192])/255.0

def draw_plot(par_dict, Acols, mjd, phase, unc, scale=None):
    '''Makes arrow plot for a given data set.
       Returns: value of the biggest arrow and plot
par_dict - dictionary with pulsar parameters
Acols - max number of inner orbital frequencies to calculate
mjd, phase, unc - parameters of data'''
    pl=pltr.plotter(par_dict, Acols, mjd, phase, unc)
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

def coeff_plot(pl, scale=None):
    '''Makes arrow plot for a given (arrow_coeff) object.
       Takes: (arrow_coeff) object. 
       Returns: value of the biggest arrow and plot
pl - object with arrow coordinates, directions and lenghts'''    
    if scale is None:
        ar_scale=np.amax(pl.M)
    else:
        ar_scale=scale
    for k in range(0,len(pl.err_X)):
        i,j = pl.err_ix[k]
        plt.plot(pl.err_X[k]/ar_scale+j,pl.err_Y[k]/ar_scale+i, color=err_bl, lw=1, alpha=0.3)
    Q = plt.quiver(pl.X, pl.Y, pl.U, pl.V, units='x', scale=ar_scale, color=err_bl)
    plt.xlabel('Inner orbit frequencies')
    plt.ylabel('Outer orbit frequencies')
    plt.ylim(np.amin(pl.Y)-0.95,np.amax(pl.Y)+0.95)
    plt.xlim(np.amin(pl.X)-0.95,np.amax(pl.X)+0.95)
    plt.gca().set_aspect('equal')
    err_red=np.array([191, 54, 12])/255.0
    plt.figtext(0.15,0.2, r'the longest arrow = %4.3f $\mu$s'%(ar_scale), fontproperties=font0, color=err_red)
    return ar_scale
