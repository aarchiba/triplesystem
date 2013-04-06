import numpy as np

import threebody
import kepler

times = np.arange(1000)
mjdbase = 55920
o = threebody.compute_orbit(threebody.best_parameters, times)

print "#\tMJD\tasini_i\tpb_i\te_i\tom_i\tt0_i\tasini_o\tpb_o\te_o\tom_o\tt0_o\tz_i\tvz_i"
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

    t0_i = (t0_i-times[i])%pb_i + times[i]+mjdbase
    t0_0 = (t0_o-times[i])%pb_o + times[i]+mjdbase
    print "\t".join(repr(p) for p in (
        times[i]+mjdbase, 
        asini_i, pb_i, e_i, om_i, t0_i, 
        asini_o, pb_o, e_o, om_o, t0_o, 
        cm_i[2], cm_i[5]))

