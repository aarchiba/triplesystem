import inspect
import numpy as np
from scipy.optimize import newton
from scipy.linalg import block_diag
import scipy.linalg

# units are days, light-seconds, and solar masses
G_mks = 6.67398e-11 # m**3 kg**(-1) s**(-2)
c = 299792458. # m/s
M_sun = 1.9891e30 # kg
G_old = G_mks * c**(-3) * M_sun * 86400**2
G = 36768.59290949113 # Based on standard gravitational parameter

def true_from_eccentric(e, eccentric_anomaly):
    """Compute the true anomaly from the eccentric anomaly

    Inputs:
        e - the eccentricity
        eccentric_anomaly - the eccentric anomaul

    Outputs:
        true_anomaly - the true anomaly
        true_anomaly_de - derivative of true anomaly with respect to e
        true_anomaly_prime - derivative of true anomaly with respect to
            eccentric anomaly
    """
    true_anomaly = 2*np.arctan2(np.sqrt(1+e)*np.sin(eccentric_anomaly/2),
                                np.sqrt(1-e)*np.cos(eccentric_anomaly/2))
    true_anomaly_de = (np.sin(eccentric_anomaly)/
            (np.sqrt(1-e**2)*(1-e*np.cos(eccentric_anomaly))))
    true_anomaly_prime = (np.sqrt(1-e**2)/(1-e*np.cos(eccentric_anomaly)))
    return true_anomaly, true_anomaly_de, true_anomaly_prime

def eccentric_from_mean(e, mean_anomaly):
    """Compute the eccentric anomaly from the mean anomaly

    Inputs:
        e - the eccentricity
        mean_anomaly - the mean anomaly

    Outputs:
        eccentric_anomaly - the true anomaly
        derivatives - pair of derivatives with respect to the two inputs
    """
    eccentric_anomaly = newton(
            lambda E: E-e*np.sin(E)-mean_anomaly,
            mean_anomaly,
            lambda E: 1-e*np.cos(E))
    eccentric_anomaly_de = np.sin(eccentric_anomaly)/(1-e*np.cos(eccentric_anomaly))
    eccentric_anomaly_prime = (1-e*np.cos(eccentric_anomaly))**(-1)
    return eccentric_anomaly, [eccentric_anomaly_de, eccentric_anomaly_prime]



def kepler_2d(a, pb, eps1, eps2, t):
    """Position and velocity of a particle in a Kepler orbit

    The orbit has semimajor axis a, period pb, and eccentricity
    paramerized by eps1=e*sin(om) and eps2=e*cos(om), and the
    particle is on the x axis at time zero, while the values
    are computed for time t.

    The function returns a pair (xv, p), where xv is of length
    four and consists of (x,y,v_x,v_y), and p is of shape (4,5)
    and cell (i,j) contains the the partial derivative of the
    ith element of xv with respect to the jth orbital parameter.

    The zero of time is when the particle is on the positive x
    axis. (Which will be the ascending node in a three-dimensional
    model.)
    """
    if eps1==0 and eps2==0:
        eps1=1e-50
    e = np.hypot(eps1, eps2)
    if e==0:
        d_e = np.array([0,0,0,0,0])
    else:
        d_e = np.array([0,0,eps1/e,eps2/e,0])
    #return e, d_e

    om = np.arctan2(eps1, eps2)
    if e==0:
        d_om = np.array([0,0,0,0,0])
    else:
        d_om = np.array([0,0,-eps1/e**2,eps2/e**2,0])
    #return om, d_om

    true_anomaly_0 = -om
    d_true_anomaly_0 = -d_om

    eccentric_anomaly_0 = np.arctan2(
            np.sqrt(1-e**2)*np.sin(true_anomaly_0),
            e + np.cos(true_anomaly_0))
    d_eccentric_anomaly_0 = (
        d_e*(-(1+e*np.cos(true_anomaly_0))*np.sin(true_anomaly_0)/
            (np.sqrt(1-e**2)*(e*np.cos(true_anomaly_0)+1)**2)) +
        d_true_anomaly_0
            *(np.sqrt(1-e**2)*(1+e*np.cos(true_anomaly_0)))
            /(e*np.cos(true_anomaly_0)+1)**2)

    mean_anomaly_0 = eccentric_anomaly_0 - e*np.sin(eccentric_anomaly_0)
    d_mean_anomaly_0 = (d_eccentric_anomaly_0
            -d_e*np.sin(eccentric_anomaly_0)
            -e*np.cos(eccentric_anomaly_0)*d_eccentric_anomaly_0)

    mean_anomaly = 2*np.pi*t/pb + mean_anomaly_0
    d_mean_anomaly = (2*np.pi*np.array([0,-t/pb**2,0,0,pb**(-1)])
            + d_mean_anomaly_0)

    mean_anomaly_dot = 2*np.pi/pb
    d_mean_anomaly_dot = 2*np.pi*np.array([0,-pb**(-2),0,0,0])
    #return [mean_anomaly, mean_anomaly_dot], [d_mean_anomaly, d_mean_anomaly_dot]
    #return mean_anomaly, d_mean_anomaly

    eccentric_anomaly, (eccentric_anomaly_de, eccentric_anomaly_prime) = eccentric_from_mean(e, mean_anomaly)
    eccentric_anomaly_dot = eccentric_anomaly_prime*mean_anomaly_dot

    d_eccentric_anomaly = (eccentric_anomaly_de*d_e
            +eccentric_anomaly_prime*d_mean_anomaly)
    d_eccentric_anomaly_prime = (np.cos(eccentric_anomaly)/(1-e*np.cos(eccentric_anomaly))**2*d_e
                -e*np.sin(eccentric_anomaly)/(1-e*np.cos(eccentric_anomaly))**2*d_eccentric_anomaly)
    d_eccentric_anomaly_dot = (d_eccentric_anomaly_prime*mean_anomaly_dot
            +eccentric_anomaly_prime*d_mean_anomaly_dot)
    #return eccentric_anomaly, d_eccentric_anomaly
    #return eccentric_anomaly_prime, d_eccentric_anomaly_prime
    #return eccentric_anomaly_dot, d_eccentric_anomaly_dot

    true_anomaly, true_anomaly_de, true_anomaly_prime = true_from_eccentric(e, eccentric_anomaly)
    true_anomaly_dot = true_anomaly_prime*eccentric_anomaly_dot

    d_true_anomaly = true_anomaly_de*d_e + true_anomaly_prime*d_eccentric_anomaly
    d_true_anomaly_prime = (
            ((np.cos(eccentric_anomaly)-e)/(np.sqrt(1-e**2)*(1-e*np.cos(eccentric_anomaly))**2))*d_e
            -e*np.sqrt(1-e**2)*np.sin(eccentric_anomaly)
             /(1-e*np.cos(eccentric_anomaly))**2*d_eccentric_anomaly)
    d_true_anomaly_dot = (d_true_anomaly_prime*eccentric_anomaly_dot
                         +true_anomaly_prime*d_eccentric_anomaly_dot)
    #return true_anomaly, d_true_anomaly
    #return true_anomaly_prime, d_true_anomaly_prime
    #return true_anomaly_dot, d_true_anomaly_dot

    r = a*(1-e**2)/(1+e*np.cos(true_anomaly))
    r_prime = (a*e*(1-e**2)*np.sin(true_anomaly)
            /(1+e*np.cos(true_anomaly))**2)
    r_dot = r_prime*true_anomaly_dot
    d_a = np.array([1,0,0,0,0])
    d_r = (d_a*r/a
          -a*d_e*((1+e**2)*np.cos(true_anomaly)+2*e)/(1+e*np.cos(true_anomaly))**2
          +r_prime*d_true_anomaly)
    d_r_prime = (d_a*r_prime/a
                +a*d_e*(-e*(1+e**2)*np.cos(true_anomaly)-3*e**2+1)*np.sin(true_anomaly)
                    /(1+e*np.cos(true_anomaly))**3
                +a*e*(1-e**2)*(e*(np.sin(true_anomaly)**2+1)+np.cos(true_anomaly))/(1+e*np.cos(true_anomaly))**3*d_true_anomaly)
    d_r_dot = d_r_prime*true_anomaly_dot + r_prime*d_true_anomaly_dot
    #return r, d_r
    #return r_prime, d_r_prime
    #return r_dot, d_r_dot

    xyv = np.zeros(4)
    xyv[0] = r*np.cos(true_anomaly+om)
    xyv[1] = r*np.sin(true_anomaly+om)
    xyv[2] = (r_dot*np.cos(true_anomaly+om)
            -r*true_anomaly_dot*np.sin(true_anomaly+om))
    xyv[3] = (r_dot*np.sin(true_anomaly+om)
            +r*true_anomaly_dot*np.cos(true_anomaly+om))

    partials = np.zeros((4, 5))

    partials[0,:] = (d_r*np.cos(true_anomaly+om)
                    -(d_true_anomaly+d_om)*r*np.sin(true_anomaly+om))
    partials[1,:] = (d_r*np.sin(true_anomaly+om)
                    +(d_true_anomaly+d_om)*r*np.cos(true_anomaly+om))
    partials[2,:] = (d_r_dot*np.cos(true_anomaly+om)
                    -(d_true_anomaly+d_om)*r_dot*np.sin(true_anomaly+om)
                    -d_r*true_anomaly_dot*np.sin(true_anomaly+om)
                    -r*d_true_anomaly_dot*np.sin(true_anomaly+om)
                    -r*true_anomaly_dot*np.cos(true_anomaly+om)
                        *(d_true_anomaly+d_om))
    partials[3,:] = (d_r_dot*np.sin(true_anomaly+om)
                    +(d_true_anomaly+d_om)*r_dot*np.cos(true_anomaly+om)
                    +d_r*true_anomaly_dot*np.cos(true_anomaly+om)
                    +r*d_true_anomaly_dot*np.cos(true_anomaly+om)
                    -r*true_anomaly_dot*np.sin(true_anomaly+om)
                        *(d_true_anomaly+d_om))

    return xyv, partials

def inverse_kepler_2d(xv,m):
    """Compute the Keplerian parameters for the osculating orbit

    No partial derivatives are computed (even though it would be much easier)
    because you can use the partials for kepler_2d and invert the matrix.
    """
    mu = G*m
    a_guess = np.hypot(xv[0],xv[1])
    h = (xv[0]*xv[3]-xv[1]*xv[2])
    r = np.hypot(xv[0],xv[1])
    eps2, eps1 = np.array([xv[3], -xv[2]])*h/mu - xv[:2]/r
    e = np.hypot(eps1, eps2)
    p = h**2/mu
    a = p/(1-e**2)
    pb = 2*np.pi*(a**3/mu)**(0.5)

    om = np.arctan2(eps1,eps2)
    true_anomaly = np.arctan2(xv[1],xv[0])-om
    eccentric_anomaly = np.arctan2(np.sqrt(1-e**2)*np.sin(true_anomaly),
                                   e+np.cos(true_anomaly))
    mean_anomaly = eccentric_anomaly - e*np.sin(eccentric_anomaly)

    true_anomaly_0 = -om
    eccentric_anomaly_0 = np.arctan2(np.sqrt(1-e**2)*np.sin(true_anomaly_0),
                                   e+np.cos(true_anomaly_0))
    mean_anomaly_0 = eccentric_anomaly_0 - e*np.sin(eccentric_anomaly_0)

    return a, pb, eps1, eps2, (mean_anomaly-mean_anomaly_0)*pb/(2*np.pi)
    #mean_anomaly*pb/(2*np.pi)

def btx_parameters(asini, pb, eps1, eps2, tasc):
    """Attempt to convert parameters from ELL1 to BTX"""
    e = np.hypot(eps1,eps2)
    om = np.arctan2(eps1,eps2)
    true_anomaly = -om # True anomaly at the ascending node
    eccentric_anomaly = np.arctan2(np.sqrt(1-e**2)*np.sin(true_anomaly),
                                   e+np.cos(true_anomaly))
    mean_anomaly = eccentric_anomaly - e*np.sin(eccentric_anomaly)
    t0 = tasc-mean_anomaly*pb/(2*np.pi)
    return asini, pb, e, om, t0

def mass(a, pb):
    """Compute the mass of a particle in a Kepler orbit

    The units are a in light seconds, binary period in seconds,
    and mass in solar masses.
    """
    return 4*np.pi**2*a**3*pb**(-2)/G
def mass_partials(a, pb):
    """Compute the mass of a particle in a Kepler orbit, with partials

    The units are a in light seconds, binary period in seconds,
    and mass in solar masses.
    """
    m = mass(a,pb)
    return m, np.array([3*m/a,-2*m/pb])


def kepler_3d(a,pb,eps1,eps2,i,lan,t):
    """One-body Kepler problem in 3D

    This function simply uses kepler_2d and rotates it into 3D.
    """
    xv, jac = kepler_2d(a,pb,eps1,eps2,t)
    xyv = np.zeros(6)
    xyv[:2] = xv[:2]
    xyv[3:5] = xv[2:]

    jac2 = np.zeros((6,7))
    t = np.zeros((6,5))
    t[:2] = jac[:2]
    t[3:5] = jac[2:]
    jac2[:,:4] = t[:,:4]
    jac2[:,-1] = t[:,-1]

    r_i = np.array([[1,0,0],
                    [0,np.cos(i),-np.sin(i)],
                    [0,np.sin(i), np.cos(i)]])
    d_r_i = np.array([[0,0,0],
                      [0,-np.sin(i),-np.cos(i)],
                      [0, np.cos(i),-np.sin(i)]])
    r_i_6 = block_diag(r_i,r_i)
    d_r_i_6 = block_diag(d_r_i,d_r_i)
    xyv3 = np.dot(r_i_6,xyv)
    jac3 = np.dot(r_i_6,jac2)
    jac3[:,4] += np.dot(d_r_i_6, xyv)

    r_lan = np.array([[ np.cos(lan),np.sin(lan),0],
                      [-np.sin(lan),np.cos(lan),0],
                      [0,0,1]])
    d_r_lan = np.array([[-np.sin(lan), np.cos(lan),0],
                        [-np.cos(lan),-np.sin(lan),0],
                        [0,0,0]])
    r_lan_6 = block_diag(r_lan,r_lan)
    d_r_lan_6 = block_diag(d_r_lan,d_r_lan)
    xyv4 = np.dot(r_lan_6,xyv3)
    jac4 = np.dot(r_lan_6,jac3)
    jac4[:,5] += np.dot(d_r_lan_6, xyv3)

    return xyv4, jac4

def inverse_kepler_3d(xyv, m):
    """Inverse Kepler one-body calculation
    """
    L = np.cross(xyv[:3],xyv[3:])
    i = np.arccos(L[2]/np.sqrt(np.dot(L,L)))
    lan = (-np.arctan2(L[0],-L[1])) % (2*np.pi)

    r_lan = np.array([[ np.cos(lan),np.sin(lan),0],
                      [-np.sin(lan),np.cos(lan),0],
                      [0,0,1]])
    r_lan_6 = block_diag(r_lan,r_lan)
    xyv2 = np.dot(r_lan_6.T,xyv)

    r_i = np.array([[1,0,0],
                    [0,np.cos(i),-np.sin(i)],
                    [0,np.sin(i), np.cos(i)]])
    r_i_6 = block_diag(r_i,r_i)
    xyv3 = np.dot(r_i_6.T,xyv2)

    xv = xyv3[np.array([True,True,False,True,True,False])]
    a,pb,eps1,eps2,t = inverse_kepler_2d(xv,m)

    return a,pb,eps1,eps2,i,lan,t


# Two body forward and back
# Additional parameters: mass_ratio, cm_x[3], cm_v[3]

def kepler_two_body(a,pb,eps1,eps2,i,lan,q,x_cm,v_cm,tasc):
    """Set up two bodies in a Keplerian orbit

    Most orbital parameters describe the orbit of the
    primary; the secondary's parameters are inferred
    from the fact that its mass is q times that of the
    primary. x_cm and v_cm are the position and velocity
    of the center of mass of the system.

    The system is observed at time zero, and tasc is the
    the time of the ascending node.

    Includes derivatives.
    """
    e = np.eye(14)
    (d_a, d_pb, d_eps1, d_eps2, d_i, d_lan, d_q) = e[:7]
    d_x_cm = e[7:10]
    d_v_cm = e[10:13]
    d_tasc = e[13]

    a_c = a/q
    a_tot = a+a_c
    d_a_c = d_a/q-a*d_q/q**2
    d_a_tot = d_a + d_a_c

    m_tot, m_tot_prime = mass_partials(a_tot, pb)
    m = m_tot/(1+q)
    m_c = q*m
    d_m_tot = (m_tot_prime[0]*d_a_tot
              +m_tot_prime[1]*d_pb)
    d_m = d_m_tot/(1+q) - m_tot*d_q/(1+q)**2
    d_m_c = d_q*m + q*d_m

    xv_tot, jac_one = kepler_3d(a_tot,pb,eps1,eps2,i,lan,-tasc)
    d_xv_tot = np.dot(jac_one,
            np.array([d_a_tot,
                      d_pb,
                      d_eps1,
                      d_eps2,
                      d_i,
                      d_lan,
                      -d_tasc]))

    xv = xv_tot/(1+1./q)
    d_xv = d_xv_tot/(1+1./q) + xv_tot[:,None]*d_q[None,:]/(1+q)**2

    xv_c = -xv/q
    d_xv_c = -d_xv/q+xv[:,None]*d_q[None,:]/q**2

    xv[:3] += x_cm # FIXME: when, if t is actually t0?
    xv[3:] += v_cm
    xv_c[:3] += x_cm
    xv_c[3:] += v_cm
    d_xv[:3] += d_x_cm
    d_xv[3:] += d_v_cm
    d_xv_c[:3] += d_x_cm
    d_xv_c[3:] += d_v_cm

    total_state = np.zeros(14)
    total_state[:6] = xv
    total_state[6] = m
    total_state[7:13] = xv_c
    total_state[13] = m_c
    d_total_state = np.zeros((14,14))
    d_total_state[:6] = d_xv
    d_total_state[6] = d_m
    d_total_state[7:13] = d_xv_c
    d_total_state[13] = d_m_c

    return total_state, d_total_state

def inverse_kepler_two_body(total_state):
    x_p = total_state[:3]
    v_p = total_state[3:6]
    m_p = total_state[6]

    x_c = total_state[7:10]
    v_c = total_state[10:13]
    m_c = total_state[13]

    x_cm = (m_p*x_p+m_c*x_c)/(m_c+m_p)
    v_cm = (m_p*v_p+m_c*v_c)/(m_c+m_p)

    x = x_p-x_c
    v = v_p-v_c

    xv = np.concatenate((x,v))

    a_tot,pb,eps1,eps2,i,lan,t = inverse_kepler_3d(xv,m_c+m_p)
    q = m_c/m_p
    a = a_tot/(1+1./q)

    return a,pb,eps1,eps2,i,lan,q,x_cm,v_cm,-t

# Three body forward and back
# Write as a two-body whose center of mass position and
# velocity are in a two-body solution with the third

def format_args(*args):
    return "\t".join([("%s:\t%s" % (n,a))
        for n,a in zip(three_body_parameters, args)])
def kepler_three_body(
    a_i,pb_i,eps1_i,eps2_i,i_i,lan_i,q_i,tasc_i,
    a_o,pb_o,eps1_o,eps2_o,i_o,lan_o,tasc_o,
    x_cm,v_cm):
    """Compute positions and velocities of three bodies

    The inner system is treated as if it were a Keplerian
    two-body system; the outer system is treated as a two
    body system as well, with one of the bodies having the
    total mass of the inner system, and located at the
    center of mass of the inner system.

    Most orbital parameters are the same as elsewhere
    in this module, suffixed only with _i or _o to
    indicate inner and outer orbits respectively. The
    exception is tasc_i and tasc_o. These are the times of
    ascending node of the inner and outer systems; the
    system state vectors are evaluated at time zero.

    """
    args = (a_i,pb_i,eps1_i,eps2_i,i_i,lan_i,q_i,tasc_i,
        a_o,pb_o,eps1_o,eps2_o,i_o,lan_o,tasc_o,
        x_cm,v_cm)

    e = np.eye(21)
    (d_a_i,d_pb_i,d_eps1_i,d_eps2_i,d_i_i,d_lan_i,d_q_i,d_tasc_i,
     d_a_o,d_pb_o,d_eps1_o,d_eps2_o,d_i_o,d_lan_o,d_tasc_o
     ) = e[:-6]
    d_x_cm = e[-6:-3]
    d_v_cm = e[-3:]
    m_i, m_i_prime = mass_partials(a_i*(1+1./q_i),pb_i)
    m_temp, m_temp_prime = mass_partials(a_o,pb_o)
    if m_i<=0:
        raise ValueError("m_i not positive: %s" % format_args(*args))
    if m_temp<=0:
        raise ValueError("m_temp not positive: %s" % format_args(*args))
    q_o = np.exp(newton(
        lambda lnq: 3*lnq-np.log1p(np.exp(lnq))*2
                - np.log(m_temp) + np.log(m_i),
        0))

    d_m_i = (m_i_prime[0]*(d_a_i*(1+1./q_i)-a_i*d_q_i/q_i**2)
            +m_i_prime[1]*d_pb_i)
    d_m_temp = (m_temp_prime[0]*d_a_o+m_temp_prime[1]*d_pb_o)
    d_q_o = -((q_o+1)**3/(q_o**2*(q_o+3))
            *(-d_m_temp/m_i+m_temp*d_m_i/m_i**2))

    if q_o<0:
        raise ValueError("Orbital parameters require outer mass to be negative: total mass is %g times inner mass" % (m_tot/m_i))

    total_state_o, jac_o = kepler_two_body(
        a_o,pb_o,eps1_o,eps2_o,i_o,lan_o,q_o,
        x_cm, v_cm,
        tasc_o)
    m = np.vstack([d_a_o,d_pb_o,
                d_eps1_o,d_eps2_o,
                d_i_o,d_lan_o,d_q_o,
                d_x_cm, d_v_cm,
                d_tasc_o])
    d_total_state_o = np.dot(jac_o,m)
    # Note: this fails because of roundoff
    #return total_state_o[j], d_total_state_o[j]

    total_state_i, jac_i = kepler_two_body(
        a_i,pb_i,eps1_i,eps2_i,i_i,lan_i,q_i,
        total_state_o[:3], total_state_o[3:6],
        tasc_i)
    d_total_state_i = np.dot(jac_i,
            np.vstack([d_a_i,d_pb_i,
                d_eps1_i,d_eps2_i,
                d_i_i,d_lan_i,d_q_i,
                d_total_state_o[:3], d_total_state_o[3:6],
                d_tasc_i]))

    total_state = np.zeros(21)
    d_total_state = np.zeros((21,21))
    total_state[:14] = total_state_i
    total_state[-7:] = total_state_o[-7:]
    d_total_state[:14] = d_total_state_i
    d_total_state[-7:] = d_total_state_o[-7:]

    return total_state, d_total_state
three_body_parameters = inspect.getargspec(kepler_three_body)[0]

def inverse_kepler_three_body(total_state):
    (a_i, pb_i, eps1_i, eps2_i,
        i_i, lan_i, q_i,
        x_cm_i, v_cm_i, tasc_i) = inverse_kepler_two_body(total_state[:14])

    m_p = total_state[6]
    m_c = total_state[13]
    m_o = total_state[20]

    xv_p = total_state[:6]
    xv_c = total_state[7:13]
    xv_o = total_state[14:20]

    ts_o = np.zeros(14)
    ts_o[:3] = x_cm_i
    ts_o[3:6] = v_cm_i
    ts_o[6] = m_p+m_c
    ts_o[7:14] = total_state[14:21]

    (a_o, pb_o, eps1_o, eps2_o,
        i_o, lan_o, q_o,
        x_cm, v_cm, tasc_o) = inverse_kepler_two_body(ts_o)

    return (
        a_i,pb_i,eps1_i,eps2_i,i_i,lan_i,q_i,tasc_i,
        a_o,pb_o,eps1_o,eps2_o,i_o,lan_o,tasc_o,
        x_cm,v_cm)


def kepler_three_body_measurable(
        # Measurable fairly directly
        asini_i, pb_i, eps1_i, eps2_i, tasc_i,
        # Not measurable apart from interaction
        acosi_i, q_i,
        # Measurable fairly directly
        asini_o, pb_o, eps1_o, eps2_o, tasc_o,
        # Not measurable apart from interaction
        acosi_o, delta_lan,
        # Not measurable at all
        lan_i, x_cm, v_cm,
        t):

    e = np.eye(22)
    (d_asini_i, d_pb_i, d_eps1_i, d_eps2_i, d_tasc_i,
        d_acosi_i, d_q_i,
        d_asini_o, d_pb_o, d_eps1_o, d_eps2_o, d_tasc_o,
        d_acosi_o, d_delta_lan,
        d_lan_i) = e[:15]
    d_x_cm = e[15:18]
    d_v_cm = e[18:21]
    d_t = e[21]

    a_i = np.hypot(asini_i, acosi_i)
    i_i = np.arctan2(asini_i, acosi_i)
    d_a_i = (d_asini_i*asini_i + d_acosi_i*acosi_i)/a_i
    d_i_i = (acosi_i*d_asini_i-asini_i*d_acosi_i)/a_i**2


    a_o = np.hypot(asini_o, acosi_o)
    i_o = np.arctan2(asini_o, acosi_o)
    d_a_o = (d_asini_o*asini_o + d_acosi_o*acosi_o)/a_o
    d_i_o = (acosi_o*d_asini_o-asini_o*d_acosi_o)/a_o**2

    lan_o = delta_lan + lan_i
    d_lan_o = d_delta_lan + d_lan_i

    j1 = np.vstack([
        d_a_i,d_pb_i,d_eps1_i,d_eps2_i,d_i_i,d_lan_i,d_q_i,d_tasc_i-d_t,
        d_a_o,d_pb_o,d_eps1_o,d_eps2_o,d_i_o,d_lan_o,d_tasc_o-d_t,
        d_x_cm,d_v_cm])

    if False:
        return ((a_i,pb_i,eps1_i,eps2_i,i_i,lan_i,q_i,tasc_i-t,
            a_o,pb_o,eps1_o,eps2_o,i_o,lan_o,tasc_o-t,
            x_cm[0],x_cm[1],x_cm[2],v_cm[0],v_cm[1],v_cm[2]),
            j1)
    state, jac = kepler_three_body(
        a_i,pb_i,eps1_i,eps2_i,i_i,lan_i,q_i,tasc_i-t,
        a_o,pb_o,eps1_o,eps2_o,i_o,lan_o,tasc_o-t,
        x_cm,v_cm)

    return state, np.dot(jac, j1)
three_body_parameters_measurable = inspect.getargspec(kepler_three_body_measurable)[0]

def inverse_kepler_three_body_measurable(state, t):
    (a_i,pb_i,eps1_i,eps2_i,i_i,lan_i,q_i,tasc_i,
        a_o,pb_o,eps1_o,eps2_o,i_o,lan_o,tasc_o,
        x_cm,v_cm) = inverse_kepler_three_body(state)

    asini_i = a_i*np.sin(i_i)
    acosi_i = a_i*np.cos(i_i)
    asini_o = a_o*np.sin(i_o)
    acosi_o = a_o*np.cos(i_o)
    delta_lan = lan_o - lan_i
    tasc_i = tasc_i + t
    tasc_o = tasc_o + t

    return (asini_i, pb_i, eps1_i, eps2_i, tasc_i,
        acosi_i, q_i,
        asini_o, pb_o, eps1_o, eps2_o, tasc_o,
        acosi_o, delta_lan,
        lan_i, x_cm, v_cm)

def numeric_partial(f, args, ix=0, delta=1e-6):
    #r = np.array(f(*args))
    args2 = list(args)
    args2[ix] = args[ix]+delta/2.
    r2 = np.array(f(*args2))
    args3 = list(args)
    args3[ix] = args[ix]-delta/2.
    r3 = np.array(f(*args3))
    return (r2-r3)/delta

def numeric_partials(f, args, delta=1e-6):
    r = [numeric_partial(f, args, i, delta) for i in range(len(args))]
    return np.array(r).T

def check_all_partials(f, args, delta=1e-6, atol=1e-4, rtol=1e-4):
    _, jac = f(*args)
    jac = np.asarray(jac)
    njac = numeric_partials(lambda *args: f(*args)[0], args, delta)

    try:
        np.testing.assert_allclose(jac, njac, atol=atol, rtol=rtol)
    except AssertionError:
        #print jac
        #print njac
        d = np.abs(jac-njac)/(atol+rtol*np.abs(njac))
        print "fail fraction:", np.sum(d>1)/float(np.sum(d>=0))
        worst_ix = np.unravel_index(np.argmax(d.reshape((-1,))),d.shape)
        print "max fail:", np.amax(d), "at", worst_ix
        print "jac there:", jac[worst_ix], "njac there:", njac[worst_ix]
        raise

def total_energy(total_state):
    total_state = np.asarray(total_state)
    if len(total_state.shape)>1:
        # recurse down one dimension at a time
        return np.array([total_energy(ts) for ts in total_state])
    n = len(total_state)/7
    e = 0
    for i in range(n):
        state_i = total_state[7*i:7*i+7]
        x_i = state_i[:3]
        v_i = state_i[3:6]
        m_i = state_i[6]

        e += 0.5*m_i*np.sum(v_i**2)

        for j in range(i):
            state_j = total_state[7*j:7*j+7]
            x_j = state_j[:3]
            v_j = state_j[3:6]
            m_j = state_j[6]

            r = np.sqrt(np.sum((x_i-x_j)**2))
            e -= G*m_i*m_j/r
    return e

def accelerations(total_state):
    n = len(total_state)/7
    a = np.zeros(3*n)
    d_a = np.zeros((3*n, 7*n))
    e = np.eye(7*n)
    for i in range(1,n):
        state_i = total_state[7*i:7*i+7]
        x_i = state_i[:3]
        m_i = state_i[6]
        d_x_i = e[7*i:7*i+3]
        d_m_i = e[7*i+6]
        for j in range(i):
            state_j = total_state[7*j:7*j+7]
            x_j = state_j[:3]
            m_j = state_j[6]
            d_x_j = e[7*j:7*j+3]
            d_m_j = e[7*j+6]

            dx = x_j-x_i
            d_dx = d_x_j - d_x_i
            r2 = np.sum(dx**2)
            d_r2 = 2*np.sum(dx[:,None]*d_dx,axis=0)

            c = G*dx*r2**(-3./2)
            d_c = G*(d_dx*r2**(-3./2)
                    -(3./2)*dx[:,None]*r2**(-5./2)*d_r2[None,:])
            a_ij = m_j*c
            a_ji = -m_i*c
            d_a_ij = (d_m_j[None,:]*c[:,None]+m_j*d_c)
            d_a_ji = -(d_m_i[None,:]*c[:,None]+m_i*d_c)

            a[3*i:3*i+3] += a_ij
            a[3*j:3*j+3] += a_ji
            d_a[3*i:3*i+3] += d_a_ij
            d_a[3*j:3*j+3] += d_a_ji
    return a, d_a

def rhs_three_body(t,state):
    n = len(state)/7
    assert n==2 or n==3
    accel = accelerations(state)[0]
    r = np.zeros_like(state)
    for i in range(n):
        r[7*i:7*i+3] = state[7*i+3:7*i+6]
        r[7*i+3:7*i+6] = accel[3*i:3*i+3]
    return r
def rhs_three_body_vectors(t,state):
    n_b = 3
    n_v = len(state)/21-1
    accel, jac = accelerations(state[:21])
    r = np.zeros_like(state)
    for i in range(n_b):
        r[7*i:7*i+3] = state[7*i+3:7*i+6]
        r[7*i+3:7*i+6] = accel[3*i:3*i+3]
    vecs = np.reshape(state[21:], (n_v, 21)).T
    #print state.shape, accel.shape, jac.shape, vecs.shape
    a_vecs = np.dot(jac, vecs)
    #assert False
    for j in range(n_v):
        for i in range(n_b):
            r[21*(j+1)+7*i:21*(j+1)+7*i+3] = state[21*(j+1)+7*i+3:21*(j+1)+7*i+6]
            r[21*(j+1)+7*i+3:21*(j+1)+7*i+6] = a_vecs[3*i:3*i+3,j]
    return r

def pack_three_body(*args):
    r = list(args)[:-2]+list(args[-2])+list(args[-1])
    return np.array(r)
def unpack_three_body(parameters):
    x_cm, v_cm = parameters[-6:-3], parameters[-3:]
    return list(parameters[:-6])+[x_cm,v_cm]

def rhs_variation_of_parameters(t,parameters):
    state, jac = kepler_three_body(*unpack_three_body(parameters))

    m_1, m_2, m_3 = state[6], state[13], state[20]
    m_i = m_1+m_2
    x_1, x_2, x_3 = state[:3], state[7:10], state[14:17]
    x_i = (m_1*x_1+m_2*x_2)/m_i

    x_3_i = x_3-x_i
    r2_3_i = np.sum(x_3_i**2)
    a_i = G*m_3*(x_3-x_i)*r2_3_i**(-3./2)

    x_3_1 = x_3-x_1
    r2_3_1 = np.sum(x_3_1**2)
    a_1 = G*m_3*(x_3-x_1)*r2_3_1**(-3./2)

    x_3_2 = x_3-x_2
    r2_3_2 = np.sum(x_3_2**2)
    a_2 = G*m_3*(x_3-x_2)*r2_3_2**(-3./2)

    # FIXME: compute these differences without subtraction
    a_1_delta = a_1-a_i
    a_2_delta = a_2-a_i

    a_3_i = -G*m_i*(x_3-x_i)*r2_3_i**(-3./2)
    a_3_1 = -G*m_1*(x_3-x_1)*r2_3_1**(-3./2)
    a_3_2 = -G*m_2*(x_3-x_2)*r2_3_2**(-3./2)

    a_3_delta = a_3_1+a_3_2-a_3_i

    extra_rhs = np.zeros(21)

    extra_rhs[3:6] = a_1_delta
    extra_rhs[10:13] = a_2_delta
    extra_rhs[17:20] = a_3_delta

    rhs = scipy.linalg.solve(jac,extra_rhs)

    rhs[7] += -1
    rhs[14] += -1

    return rhs

def rhs_variation_of_parameters_stabilized(t,parameters):
    state, jac = kepler_three_body(*unpack_three_body(parameters))

    m_1, m_2, m_3 = [np.float128(m) for m in [state[6], state[13], state[20]]]
    m_i = m_1+m_2
    x_1, x_2, x_3 = [x.astype(np.float128)
            for x in [state[:3], state[7:10], state[14:17]]]
    x_i = (m_1*x_1+m_2*x_2)/m_i

    x_3_i = x_3-x_i
    r2_3_i = np.sum(x_3_i**2)
    a_i = G*m_3*(x_3-x_i)*r2_3_i**np.float128(-3./2)

    x_3_1 = x_3-x_1
    r2_3_1 = np.sum(x_3_1**2)
    a_1 = G*m_3*(x_3-x_1)*r2_3_1**np.float128(-3./2)

    x_3_2 = x_3-x_2
    r2_3_2 = np.sum(x_3_2**2)
    a_2 = G*m_3*(x_3-x_2)*r2_3_2**np.float128(-3./2)

    # FIXME: compute these differences without subtraction
    a_1_delta = a_1-a_i
    a_2_delta = a_2-a_i

    a_3_i = -G*m_i*(x_3-x_i)*r2_3_i**np.float128(-3./2)
    a_3_1 = -G*m_1*(x_3-x_1)*r2_3_1**np.float128(-3./2)
    a_3_2 = -G*m_2*(x_3-x_2)*r2_3_2**np.float128(-3./2)

    a_3_delta = a_3_1+a_3_2-a_3_i

    extra_rhs = np.zeros(21)

    extra_rhs[3:6] = a_1_delta
    extra_rhs[10:13] = a_2_delta
    extra_rhs[17:20] = a_3_delta

    rhs = scipy.linalg.solve(jac,extra_rhs)

    rhs[6] = 0
    rhs[7] += -1
    rhs[14] += -1
    rhs[-6:] = 0

    return rhs
