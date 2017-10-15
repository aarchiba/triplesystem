from __future__ import division, print_function

import numpy as np
import scipy.integrate

pi = np.pi

c = 3e10            # cgs
G = 6.67408e-8      # cgs units; from wikipedia
rho_nuc = 2e14      # nuclear density; just a scaling
m_b = 1.66e-24      # g; neutron mass from de96
solar_mass = 1.98855e33  # g
n_0 = 0.1           # fm^{-3}; from de96

# EOS from Haensel et al. 1981, EOS ".20"
# n (fm^-3), rho (g cm^-3) P (erg cm^-3)
table_20 = [
    [0.1, 0.1686e15, 0.8601e33],
    [0.2, 0.3412e15, 0.1190e35],
    [0.3, 0.5267e15, 0.5390e35],
    [0.4, 0.7353e15, 0.1394e36],
    [0.5, 0.9733e15, 0.2664e36],
    [0.6, 0.1244e16, 0.4304e36],
    [0.8, 0.1886e16, 0.8584e36],
    [1.0, 0.2664e16, 0.1411e37],
    [1.2, 0.3579e16, 0.2084e37],
    [1.4, 0.4630e16, 0.2875e37],
    [1.6, 0.5816e16, 0.3783e37],
    [1.8, 0.7137e16, 0.4807e37],
    [2.0, 0.8591e16, 0.5947e37],
    [2.2, 0.1018e17, 0.7201e37],
    [2.4, 0.1190e17, 0.8571e37],
    [2.6, 0.1375e17, 0.1005e38],
    [2.8, 0.1573e17, 0.1165e38],
    [3.0, 0.1785e17, 0.1337e38],
]
# Table from Baym et al. 1971; used for low densities
# n (fm^-3), rho (g cm^-3) P (erg cm^-3)
table_baym = [
 [6.294999999999999e-12, 10440.0, 9.744e+18],
 [1.5809999999999998e-11, 26220.0, 4.968e+19],
 [3.972e-11, 65870.0, 2.431e+20],
 [9.976e-11, 165400.0, 1.151e+21],
 [2.506e-10, 415600.0, 5.266e+21],
 [6.293999999999999e-10, 1044000.0, 2.318e+22],
 [1.5809999999999998e-09, 2622000.0, 9.755e+22],
 [3.972e-09, 6588000.0, 3.911e+23],
 [4.999999999999999e-09, 8293000.0, 5.259e+23],
 [9.976e-09, 16550000.0, 1.435e+24],
 [1.99e-08, 33020000.0, 3.833e+24],
 [3.972e-08, 65890000.0, 1.006e+25],
 [7.923999999999999e-08, 131500000.0, 2.604e+25],
 [1.581e-07, 262400000.0, 6.676e+25],
 [1.99e-07, 330400000.0, 8.738e+25],
 [3.155e-07, 523700000.0, 1.629e+26],
 [5e-07, 830100000.0, 3.029e+26],
 [6.294e-07, 1045000000.0, 4.129e+26],
 [7.924e-07, 1316000000.0, 5.036e+26],
 [9.976e-07, 1657000000.0, 6.86e+26],
 [1.581e-06, 2626000000.0, 1.272e+27],
 [2.506e-06, 4164000000.0, 2.356e+27],
 [3.972e-06, 6601000000.0, 4.362e+27],
 [4.9999999999999996e-06, 8312000000.0, 5.662e+27],
 [6.293999999999999e-06, 10460000000.0, 7.702e+27],
 [7.923999999999999e-06, 13180000000.0, 1.048e+28],
 [9.975999999999999e-06, 16590000000.0, 1.425e+28],
 [1.2559999999999998e-05, 20900000000.0, 1.938e+28],
 [1.581e-05, 26310000000.0, 2.503e+28],
 [1.99e-05, 33130000000.0, 3.404e+28],
 [2.5059999999999997e-05, 41720000000.0, 4.628e+28],
 [3.155e-05, 52540000000.0, 5.949e+28],
 [3.972e-05, 66170000000.0, 8.089e+28],
 [4.9999999999999996e-05, 83320000000.0, 1.1e+29],
 [6.293999999999999e-05, 104900000000.0, 1.495e+29],
 [7.924e-05, 132200000000.0, 2.033e+29],
 [9.976e-05, 166400000000.0, 2.597e+29],
 [0.00011049999999999999, 184400000000.0, 2.892e+29],
 [0.0001256, 209600000000.0, 3.29e+29],
 [0.0001581, 264000000000.0, 4.473e+29],
 [0.000199, 332500000000.0, 5.816e+29],
 [0.00025059999999999997, 418800000000.0, 7.538e+29],
 [0.0002572, 429900000000.0, 7.805e+29],
 [0.000267, 446000000000.0, 7.89e+29],
 [0.00031259999999999995, 522800000000.0, 8.352e+29],
 [0.00039509999999999995, 661000000000.0, 9.098e+29],
 [0.0004759, 796400000000.0, 9.831e+29],
 [0.0005812, 972800000000.0, 1.083e+30],
 [0.0007143, 1196000000000.0, 1.218e+30],
 [0.0008785999999999999, 1471000000000.0, 1.399e+30],
 [0.001077, 1805000000000.0, 1.638e+30],
 [0.0013139999999999998, 2202000000000.0, 1.95e+30],
 [0.001748, 2930000000000.0, 2.592e+30],
 [0.002287, 3833000000000.0, 3.506e+30],
 [0.0029419999999999997, 4933000000000.0, 4.771e+30],
 [0.0037259999999999997, 6248000000000.0, 6.481e+30],
 [0.00465, 7801000000000.0, 8.748e+30],
 [0.005728, 9611000000000.0, 1.17e+31],
 [0.007424, 12460000000000.0, 1.695e+31],
 [0.008907, 14960000000000.0, 2.209e+31],
 [0.01059, 17780000000000.0, 2.848e+31],
 [0.01315, 22100000000000.0, 3.931e+31],
 [0.01777, 29880000000000.0, 6.178e+31],
 [0.022389999999999997, 37670000000000.0, 8.774e+31],
 [0.030169999999999995, 50810000000000.0, 1.386e+32],
 [0.03675, 61930000000000.0, 1.882e+32],
 [0.04585, 77320000000000.0, 2.662e+32],
 [0.05821, 98260000000000.0, 3.897e+32],
 [0.07468, 126200000000000.0, 5.861e+32],
 [0.09370999999999999, 158600000000000.0, 8.595e+32],
 [0.1182, 200400000000000.0, 1.286e+33],
 [0.1484, 252000000000000.0, 1.9e+33],
 [0.1625, 276100000000000.0, 2.242e+33],
 [0.18139999999999998, 308500000000000.0, 2.751e+33],
 [0.2017, 343300000000000.0, 3.369e+33],
 [0.22799999999999998, 388500000000000.0, 4.286e+33],
 [0.27149999999999996, 463600000000000.0, 6.103e+33],
 [0.2979, 509400000000000.0, 7.391e+33]
]
# table from Malone et al. 1975
# n (fm^-3), rho (g cm^-3) P (erg cm^-3)
table_malone = [
    [0.1, 1.68889e+14, 1.74e+33],
    [0.15, 2.54444e+14, 4.55e+33],
]

table_merged = ([[n, rho, P] for (n, rho, P) in table_baym if n <= 0.1]
                + [[n, rho, P] for (n, rho, P) in table_20 if n > 0.1])
table_merged = np.array(table_merged)

if np.any(np.diff(table_merged, axis=0) < 0):
    raise ValueError

def lookup(table_x, table_y, x):
    if len(table_x) != len(table_y):
        raise ValueError
    if x <= 0:
        return 0
    i = np.searchsorted(table_x, x)
    if i == 0:
        i += 1
    if i == len(table_x):
        i -= 1
    x0, x1 = table_x[i-1:i+1]
    y0, y1 = table_y[i-1:i+1]
    t = (np.log(x)-np.log(x0))/(np.log(x1)-np.log(x0))
    return np.exp(np.log(y0)+t*(np.log(y1)-np.log(y0)))
    

def eos_20_rho_p(p):
    return lookup(table_merged[:,2], table_merged[:,1], p)

    
def eos_20_n_p(p):
    return lookup(table_merged[:,2], table_merged[:,0], p)


de96_Gamma = 2.34
de96_K = 0.0195
def eos_de96_rho_n(n):
    return n*m_b + de96_K*n_0*m_b/(de96_Gamma-1)*(n/n_0)**de96_Gamma
def eos_de96_p_n(n):
    return de96_K*n_0*m_b*(n/n_0)**de96_Gamma
def eos_de96_n_p(p):
    return n_0*(p/(de96_K*n_0*m_b))**(1/de96_Gamma)
def eos_de96_rho_p(p):
    return eos_de96_rho_n(eos_de96_n_p(p))

class NeutronStar(object):

    def __init__(self, a, b, rho_start=1e-6):
        self.a = a
        self.b = b
        self.rho_start = rho_start

    def A(self, phi):
        return np.exp(self.a*phi+self.b*phi**2/2)

    def energy_density(self, p):
        return eos_20_rho_p(p)*c**2

    def n(self, p):
        return eos_20_n_p(p)

    def RHS(self, rho, x):
        """RHS for EOS integration

        Based on Damour and Esposito-Farese 1996
        """

        M, nu, phi, psi, p, Mb, omega, omicron = x

        A_phi = self.A(phi)
        a_phi = self.a+self.b*phi
        e = self.energy_density(p)
        n = self.n(p)

        M_prime = (4*np.pi*G/c**4*rho**2*A_phi**4*e
                   + rho*(rho-2*M)*psi**2)
        nu_prime = (8*np.pi*G/c**4*rho**2*A_phi**4*p/(rho-2*M)
                    + rho*psi**2 + 2*M/(rho*(rho-2*M)))
        phi_prime = psi
        psi_prime = (4*np.pi*G/c**4*rho*A_phi**4/(rho-2*M)
                     * (a_phi*(e-3*p)+rho*psi*(e-p))
                     - 2*(rho-M)*psi/(rho*(rho-2*M)))
        p_prime = -(e+p)*(4*np.pi*G/c**4*rho**2*A_phi**4*p/(rho-2*M)
                          + rho*psi**2/2
                          + M/(rho*(rho-2*M))
                          + a_phi*psi)
        Mb_prime = 4*np.pi*m_b*n*A_phi**3*rho**2/np.sqrt(1-2*M/rho)
        omega_prime = omicron
        omicron_prime = (4*np.pi*G/c**4*rho**2/(rho-2*M)*A_phi**4
                         * (e+p)*(omicron*4*omega/rho)
                         + (psi**2*rho - 4/rho)*omicron)

        r = np.array([M_prime, nu_prime, phi_prime, psi_prime, 
                      p_prime, Mb_prime, omega_prime, omicron_prime])
        return r

    def setup_initial(self, p_c, phi_c):
        A_phi = self.A(phi_c)
        a_phi = self.a+self.b*phi_c
        e = self.energy_density(p_c)

        M = 0
        nu = 0
        phi = phi_c
        psi = (self.rho_start/3
               * 4*np.pi*G/c**4
               * A_phi**4
               * a_phi*(e-3*p_c))
        Mb = 0
        omega = 1
        omicron = (self.rho_start/5
                   * 16*np.pi*G/c**4*A_phi**4
                   * (e+p_c)*omega)
        return np.array([M, nu, phi, psi, p_c, Mb, omega, omicron])
               
    def match_external(self, rho, x):
        M, nu, phi, psi, p, Mb, omega, omicron = x

        self.R = rho
        self.nu_prime = self.R*psi**2+2*M/(self.R*(self.R-2*M))
        self.alpha_A = 2*psi/self.nu_prime
        self.Q1 = np.sqrt(1+self.alpha_A**2)
        self.Q2 = np.sqrt(1-2*M/self.R)
        self.nu_hat = -2/self.Q1*np.arctanh(self.Q1/(1+2/(self.R*self.nu_prime)))
        self.phi_0 = phi-self.alpha_A*self.nu_hat/2
        self.m_A = (c**2/(2*G)*self.nu_prime*self.R**2*self.Q2
                    * np.exp(self.nu_hat/2))
        self.mb_A = Mb
        self.J_A = (c**2/(6*G)*omicron*self.R**4*self.Q2
                    * np.exp(-self.nu_hat/2))
        self.Omega = omega - (c**4/G**2
                              * 3*self.J_A/(4*self.m_A**3*(3-self.alpha_A**2))
                              * (np.exp(2*self.nu_hat) - 1
                                 + (4*G*self.m_A/(self.R*c**2)
                                    * np.exp(self.nu_hat)
                                    * (2*G*self.m_A/(self.R*c**2)
                                       + np.exp(self.nu_hat/2)
                                       * np.cosh(self.Q1*self.nu_hat/2)))))
        self.I_A = self.J_A/self.Omega
        self.alpha_0 = self.a+self.b*self.phi_0
        self.beta_0 = self.b

        # Unit conversion
        self.R *= 1e-5
        self.m_A /= solar_mass
        self.mb_A *= 1e39/solar_mass

        self.Delta = self.alpha_0*(self.alpha_A-self.alpha_0)

    def integrate(self, p_c, phi_c):
        x = self.setup_initial(p_c, phi_c)
        O = scipy.integrate.ode(self.RHS)
        O.set_integrator('vode')
        O.set_initial_value(x, self.rho_start)
        p = p_c
        rn = self.rho_start
        rs = [rn]
        xs = [x]
        while p > 0 and O.successful():
            x = O.integrate(1e10, step=True)
            rn = O.t
            M, nu, phi, psi, p, Mb, omega, omicron = x
            rs.append(rn)
            xs.append(x)
        # automatic step size control senses the discontinuity and stops

        self.match_external(rn, x)

        return rs, xs


def evaluate(p_c, phi_c, beta_0):
    N = NeutronStar(a=0, b=beta_0)
    N.integrate(p_c, phi_c)
    d = dict(initial=(p_c, phi_c, beta_0))
    for k in dir(N):
        a = getattr(N, k)
        if k.startswith("_") or callable(a):
            continue
        d[k] = a
    return d


p_c_start = 1e35   # <1 M_sun in GR

def mr_curve(points, phi_c, beta_0, mass):
    points.sort()
    if not points:
        p_c = 1e34
        d = evaluate(p_c, phi_c, beta_0)
        points.append((p_c, d))

    # FIXME: these can just check the first and last
    def has_incr():
        if len(points) < 2:
            return False
        return points[0][1]["m_A"] < points[1][1]["m_A"]

    def has_decr():
        if len(points) < 2:
            return False
        return points[-2][1]["m_A"] > points[-1][1]["m_A"]

    while not has_incr() or points[0][1]["m_A"] > mass:
        p_c = points[0][0]*(0.9)
        d = evaluate(p_c, phi_c, beta_0)
        points.insert(0, (p_c, d))

    while not has_decr():
        p_c = points[-1][0]*(1.1)
        d = evaluate(p_c, phi_c, beta_0)
        points.append((p_c, d))
        
    peak = None
    while peak is None or points[peak][1]["m_A"] < mass:
        for i in range(1, len(points)-1):
            if (points[i-1][1]["m_A"] < points[i][1]["m_A"]
                and points[i][1]["m_A"] >= points[i+1][1]["m_A"]):
                peak = i
                break
        else:
            raise ValueError
    
        # FIXME: infinite loop if peak not achievable
        # Use brent to find the actual peak
        if points[peak][1]["m_A"] < mass:
            p_c = (points[peak][0]+points[peak+1][0])/2
            d = evaluate(p_c, phi_c, beta_0)
            points.insert(peak+1, (p_c, d))

            p_c = (points[peak][0]+points[peak-1][0])/2
            d = evaluate(p_c, phi_c, beta_0)
            points.insert(peak, (p_c, d))

    for i in range(1, peak+1):
        if points[i][1]["m_A"] >= mass:
            break
    else:
        raise ValueError

    if points[i][1]["m_A"] == mass:
        return points[i][1]["m_A"]
    
    a, b = points[i-1][0], points[i][0]
    pd = {p_c: d for (p_c, d) in points} 

    def f(p_c):
        try:
            d = pd[p_c]
        except KeyError:
            d = evaluate(p_c, phi_c, beta_0)
            pd[p_c] = d
        return d["m_A"] - mass

    p_c = scipy.optimize.brentq(f, a, b)
    d = pd[p_c]

    # Can't reassign points
    del points[:]
    points.extend([ i for i in pd.items() ])
    points.sort()

    return d
