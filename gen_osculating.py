import numpy as np
import scipy.linalg

import threebody
import kepler

mjds, delays, tel_list, tels, uncerts = threebody.load_data()
times = np.arange(1000)
f0, f0err = 365.9533436144258189, 0.0000000000656904
f1, f1err = 7.833539631670e-15, 1.490959143049e-17
pepoch = 56100.000000
mjdbase = 55920
o_fit = threebody.compute_orbit(threebody.best_parameters_nogr, mjds)
lin, lin_names = threebody.trend_matrix(mjds, tel_list, tels, 
        f0=f0, pepoch=pepoch, mjdbase=mjdbase)
odelay = o_fit["delays"]
ps = np.sqrt(np.mean(lin**2, axis=0))
x, rk, resids, s = scipy.linalg.lstsq(lin/uncerts[:,None]/ps[None,:],(odelay-delays)/uncerts)
lin_parameters = -x/ps

L = lin[:,[lin_names.index("const"),lin_names.index("f0error")]]
x, rk, resids, s = scipy.linalg.lstsq(L/uncerts[:,None],  
        (86400*o_fit["einstein_delays"])/uncerts)
f0_einstein = x[1]

f0 += -lin_parameters[lin_names.index("f0error")]-f0_einstein
f1 += -lin_parameters[lin_names.index("f1error")]

o = threebody.compute_orbit(threebody.best_parameters_nogr, times)

print "#\tMJD\tf0\tf1\tasini_i\tpb_i\te_i\tom_i\tt0_i\tasini_o\tpb_o\te_o\tom_o\tt0_o\tz_i\tvz_i"
for i in range(len(times)):
    params = kepler.inverse_kepler_three_body_measurable(o["states"][i,:21], times[i])
    (asini_i, pb_i, eps1_i, eps2_i, tasc_i, 
        acosi_i, q_i,
        asini_o, pb_o, eps1_o, eps2_o, tasc_o, 
        acosi_o, delta_lan,
        lan_i, x_cm, v_cm) = params

    # WARNING WARNING WARNING: kepler.py exhanges eps1 and eps2 relative to TEMPO's ELL1 model. 
    asini_i, pb_i, e_i, om_i, t0_i = kepler.btx_parameters(asini_i, pb_i, eps1_i, eps2_i, tasc_i)
    asini_o, pb_o, e_o, om_o, t0_o = kepler.btx_parameters(asini_o, pb_o, eps1_o, eps2_o, tasc_o)
    s = o["states"][i]
    cm_i = (s[:6]*s[6]+s[7:13]*s[13])/(s[6]+s[13])

    t0_i = (t0_i-times[i]+pb_i/2)%pb_i -pb_i/2 + times[i]+mjdbase
    assert abs(t0_i-(times[i]+mjdbase))<=pb_i/2
    t0_o = (t0_o-times[i]+pb_o/2)%pb_o -pb_o/2 + times[i]+mjdbase
    print "\t".join(repr(p) for p in (
        times[i]+mjdbase, f0, f1,
        asini_i, pb_i, e_i, np.degrees(om_i), t0_i, 
        asini_o, pb_o, e_o, np.degrees(om_o), t0_o, 
        cm_i[2], cm_i[5]))

