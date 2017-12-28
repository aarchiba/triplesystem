import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from collections import namedtuple
from pylab import rcParams
rcParams['savefig.dpi'] = 144
import pickle
from astropy.io import fits
import os, sys, math, time
import math
from scipy.stats import norm
import matplotlib.mlab as mlab
from scipy import stats
from scipy.stats import kstest
from scipy.stats import norm
from matplotlib.font_manager import FontProperties

def mpu(data):
    limiit=10000
    base_mjd=data['base_mjd']
    c = abs(data['residuals']/data['phase_uncerts'])<limiit
    phase = np.array(data['residuals'][c], dtype=np.float64)
    mjd = np.array(data['times'][c], dtype=np.float64)+base_mjd
    unc = np.array(data['phase_uncerts'][c], dtype=np.float64)
    return mjd, phase, unc

tel_out=namedtuple("tel_transform",
       ["phase", "mjd","unc", "index", "name"])

def tel_transform(data,name,limiit=None):
    '''takes data package and extracts residuals corresponding to particular telescope.
       gives back dictionary of phase and mjd, unc, and buller array of indexes (corresponding indexes from the common data dictionary)
data - pickle or npz package
name - name of the telescope in quotes
limiit - in case you want to exctract only high SN ratio residuals. Use this as an upper limit'''
    if name == 'all':
        tel_res=np.array(data['residuals'])
        tel_unc=np.array(data['phase_uncerts'])
        tel_mjd=np.array(data['times'])
        tel_index= np.ones((len(data['residuals'])), dtype = np.bool)
    else:
        tel_res=np.array(data['residuals'][data['telescopes']==list(data['telescope_list']).index(name)], dtype=np.float64)
        tel_unc=np.array(data['phase_uncerts'][data['telescopes']==list(data['telescope_list']).index(name)], dtype=np.float64)
        tel_mjd=np.array(data['times'][data['telescopes']==list(data['telescope_list']).index(name)], dtype=np.float64)+data['base_mjd']
        tel_index=(data['telescopes']==list(data['telescope_list']).index(name))
    
    if limiit is None:
        limm=10000
    else:
        limm=limiit
    tel_phase=tel_res[abs(tel_res/tel_unc) <limm]
    tel_mjd=tel_mjd[abs(tel_res/tel_unc) <limm]
    tel_unc=tel_unc[abs(tel_res/tel_unc) <limm]
    
    return tel_out(phase=tel_phase, mjd=tel_mjd,unc=tel_unc, index=tel_index, name=name)


import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

mjd_year = 55197
year_mjd = 2010
def day_to_year(mjd):
    return (mjd-mjd_year)/365.2425+year_mjd


def plot_res(data, mjd, phase, unc, name, color='b',scl=None):
    base_mjd=data['base_mjd']    
    bp = data["best_parameters"]
    #p_period=1./bp['f0']
    p_period = 0.00273258863228
    pb_i = bp['pb_i']
    pb_o = bp['pb_o']
    tasc_i = bp["tasc_i"]
    tasc_o = bp["tasc_o"]
    microsec=1e6*p_period
    
    if scl is None:
        scale=microsec
    else:
        scale=scl
    
    #a0_format:
    im_width=15
    my_width=15
    mar_size=my_width*0.28
    lab_size=my_width*1.4
    tick_size=my_width*0.66
    font_size=my_width*1.8    
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    clr_grey=np.array([170, 183, 184])/255.0
    for i in range(0,6):
        t = pb_o*i+tasc_o+base_mjd
        ax1.axvline(t, linestyle='--', color=clr_grey)
        
    for k in range(0,len(color)):
        tel=tel_transform(data,'%s'%name[k])
        if color[k]!='g' or color[k]!='y' or color[k]!='b' or color[k]!='r' or color[k]!='m':
            err_clr=color[k]
            face_clr=color[k]
            mec_clr=color[k]

        if color[k]=='g':
            err_clr=np.array([129, 199, 132])/255.0
            face_clr=np.array([76, 175, 80])/255.0
            mec_clr='darkgreen'

        if color[k]=='y':
            err_clr=np.array([255, 238, 88])/255.0
            face_clr=np.array([255, 235, 59])/255.0
            mec_clr=np.array([245, 127, 23])/255.0

        if color[k]=='b':
            err_clr=np.array([159, 168, 218])/255.0
            face_clr=np.array([92, 107, 192])/255.0
            mec_clr=np.array([40, 53, 147])/255.0

        if color[k]=='r':
            err_clr=np.array([236, 64, 122])/255.0
            face_clr=np.array([194, 24, 91])/255.0
            mec_clr=np.array([136, 14, 79])/255.0

        if color[k]=='m':
            err_clr=np.array([175, 122, 197])/255.0
            face_clr=np.array([142, 68, 173])/255.0
            mec_clr=np.array([74, 35, 90])/255.0
            
        ax1.errorbar(mjd[tel.index], phase[tel.index]*scale, 
                     yerr=unc[tel.index]*scale, 
                     marker="o", 
                     linestyle='none', 
                     mec=mec_clr, 
                     markerfacecolor=face_clr, 
                     color=err_clr, 
                     markersize=mar_size)

    plt.rc('xtick', labelsize=lab_size) 
    plt.rc('ytick', labelsize=lab_size) 

    majorLocator = MultipleLocator(50)
    majorFormatter = FormatStrFormatter('%.0f')
    minorLocator = MultipleLocator(10)
    yearLocator = MultipleLocator(1)

    font0 = FontProperties()
    font0.set_size('%d'%font_size)
    font0.set_family('sans')
    font0.set_style('normal')
    font0.set_weight('bold')

    ax1.set_xlabel('MJD',  fontproperties=font0)
    ax1.tick_params('x', colors='k', size=tick_size)
    ax1.tick_params('y', colors='k', size=tick_size)

    ax2.set_xlabel('Years',  fontproperties=font0)
    ax2.tick_params('x', colors='k', size=tick_size)
    ax2.xaxis.set_major_formatter(majorFormatter)
    ax1.tick_params('x', colors='k', size=tick_size)
    if scale==microsec:
        ax1.set_ylabel(r'residuals ($\mu$s)', fontproperties=font0)
    else:
        ax1.set_ylabel(r'residuals', fontproperties=font0)
    ax2.xaxis.set_major_locator(yearLocator)

    ax1.set_xlim(base_mjd,base_mjd+2010)

    d_n, d_x = ax1.get_xlim()
    ax2.set_xlim(day_to_year(d_n), day_to_year(d_x))
    print ax1.get_xlim()
    print ax2.get_xlim()
    
    plt.gcf().set_size_inches(im_width,im_width*4./15.)

    return

#Function to fit normal distribution to it and plot it
def fit_plot(emac, number, name, color, numplots, limm):
    mu, std = norm.fit(emac)
    ax=plt.subplot(2, int(numplots/2), number)
    ax.hist(emac, bins=100, normed=True, alpha=0.6, color=color)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    ax.set_xlabel(r'$\mu$ = %.4f,  $\sigma$ = %.4f'%(mu, std), fontsize=20)
    ax.set_ylabel('res/unc', fontsize=20)
    ax.set_xlim(-limm,limm)
    if number%2:
        xx=0.13
    else:
        xx=0.55
    if (number >2.5):
        yy=0.45
    else:
        yy=0.87
    plt.figtext(xx,yy, '%s'%name, fontsize=20, color=color)
    return std

def plot_hex(par_dict, mjd,res, size, colorbar=True):
    my_width=6
    mar_size=my_width*0.33
    lab_size=my_width*1.7
    tick_size=my_width*0.66
    font_size=my_width*2.0
    
    font2 = FontProperties()
    font2.set_size('%d'%font_size)
    font2.set_family('sans')
    font2.set_style('normal')
    font2.set_weight('bold')
    plt.tick_params('x', colors='k', size=tick_size)
    plt.tick_params('y', colors='k', size=tick_size)
    plt.rc('xtick', labelsize=lab_size) 
    plt.rc('ytick', labelsize=lab_size)
    

    pb_i = par_dict['pb_i']
    pb_o = par_dict['pb_o']
    plt.set_cmap('coolwarm')
    im=plt.hexbin((mjd/pb_i)%1, mjd,res,size) 
    plt.xlabel("inner phase", fontproperties=font2)
    plt.ylabel("MJD", fontproperties=font2)
    if colorbar:
        cb = plt.colorbar(im)
        cb.set_label('ns', fontproperties=font2)
    plt.gcf().set_size_inches(my_width,my_width*0.77)
    return

