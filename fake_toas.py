
import numpy as np
import scipy.linalg

import kepler
import quad_integrate
import threebody

mjds, delays, tel_list, tels, uncerts = threebody.load_data()

f0, f0err = 365.9533436144258189, 0.0000000000656904
f1, f1err = 7.833539631670e-15, 1.490959143049e-17
pepoch = 56100.000000
mjdbase = 55920

lin, lin_names = threebody.trend_matrix(mjds, tel_list, tels)

o = threebody.compute_orbit(threebody.best_parameters,mjds)
odelay = o["delays"]
ps = np.sqrt(np.mean(lin**2, axis=0))
x, rk, resids, s = scipy.linalg.lstsq(lin/uncerts[:,None]/ps[None,:],(odelay-delays)/uncerts)
lin_parameters = -x/ps

lindelay = np.dot(lin,lin_parameters)
rms = np.sqrt(np.mean((odelay+lindelay-delays)**2))
#for n,p in zip(lin_names, lin_parameters):
#    print n, repr(p)
#print np.sqrt(np.mean((odelay-delays+np.dot(lin,lin_parameters))**2))

days = 1000
n_per_day = 100
times = np.linspace(1,days+1,n_per_day*days+1)
#times = mjds

def phase(t):
    dt = (t-(pepoch-mjdbase))*86400
    return dt*f0 + 0.5*dt**2*f1
for i in range(len(times)):
    t = times[i]
    pulse = np.round(phase(t))
    dt = (t-(pepoch-mjdbase))*86400
    t -= ((phase(t)-pulse)/(f0+dt*f1))/86400
    # now t is a pulse emission time
    times[i] = t

o = threebody.compute_orbit(threebody.best_parameters,times)
lin, lin_names = threebody.trend_matrix(times, None, None, jumps=False)
lindelay = np.dot(lin,lin_parameters[:len(lin_names)])

with open("fake.tim","wt") as tim:
    with open("fake.pulses","wt") as pulses:
        for i in range(len(o["times"])):

            t = o["times"][i]
            d = o["delays"][i] + lindelay[i]
            t += d/86400
            # now t is a barycentered arrival time
            toaline = ("@             999999.999 %05d.%s%9.2f\n" %
                (mjdbase+int(np.floor(t)),
                 ("%.13f" % (t-np.floor(t)))[2:15],
                 rms*1e6))
            tim.write(toaline)
            pulses.write("%d\n" % pulse)
