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


def draw_plot(par_dict, Acols, mjd, phase, unc, scale=None, plot_unc=True, color=err_bl, units=''):
    '''Makes arrow plot for a given data set.
       Returns: value of the biggest arrow and plot
par_dict - dictionary with pulsar parameters
Acols - max number of inner orbital frequencies to calculate
mjd, phase, unc - parameters of data (in units you want to get your arrow lenghts in)'''
    pl=pltr.plotter(par_dict, Acols, mjd, phase, unc)
    return coeff_plot(pl,scale=scale, plot_unc=plot_unc, color=color, units=units)

def coeff_plot(pl, scale=None, plot_unc=True, color=err_bl, units=''):
    '''Makes arrow plot for a given (arrow_coeff) object.
       Takes: (arrow_coeff) object. 
       Returns: value of the biggest arrow in input units and draws plot
pl - object with arrow coordinates, directions and lenghts (input your units)'''
    if scale is None:
        ar_scale=np.amax(np.hypot(pl.V,pl.U))
    else:
        ar_scale=scale
    if plot_unc:
        for k in range(0,len(pl.err_X)):
            i,j = pl.err_ix[k]
            plt.plot(pl.err_X[k]/ar_scale+j,pl.err_Y[k]/ar_scale+i, color=color, lw=1, alpha=0.3)
    Q = plt.quiver(pl.X, pl.Y, pl.U, pl.V, units='x', scale=ar_scale, color=color)
    Acols=np.amax(pl.X)+1
    plt.xlim(-Acols+0.05,Acols-0.05)
    plt.ylim(-0.95,Acols-0.05)
    plt.xlabel('Inner orbit frequencies')
    plt.ylabel('Outer orbit frequencies')
    plt.gca().set_aspect('equal')
    err_red=np.array([191, 54, 12])/255.0
    plt.figtext(0.15,0.2, r'the longest arrow = %f %s'%(ar_scale,units), fontproperties=font0, color=err_red)
    return ar_scale

