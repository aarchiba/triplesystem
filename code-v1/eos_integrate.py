from __future__ import division, print_function

import numpy as np
import scipy.integrate

pi = np.pi

c = 3e10          # cgs
G = 6.67408e-8    # cgs units
rho_nuc = 2e14    # nuclear density; just a scaling

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


def eos_20(p):
    if p <= 0:
        return 0
    i = np.searchsorted(table_merged[:, 2], p)
    if i == len(table_merged):
        # off the top end, so extrapolate
        i -= 1
    p0, p1 = table_merged[i:i+2, 2]
    rho0, rho1 = table_merged[i:i+2, 1]
    t = (np.log(p)-np.log(p0))/(np.log(p1)-np.log(p0))
    return np.exp(np.log(rho0)+t*(np.log(rho1)-np.log(rho0)))


class NeutronStar(object):

    def __init__(self, a, b, r_start=10.):
        self.a = a
        self.b = b
        self.r_start = r_start

    def A(self, phi):
        return np.exp(self.a*phi+self.b*phi**2/2)

    def Rho(self, P):
        return eos_20(P)

    def RHS(self, r, x):
        """RHS for EOS integration

        Based on Horbatsch and Burgess 2011 eqns. 3.8--3.11
        """

        p, phi, omega, mu = x
        Rho_0 = self.Rho_0
        rho_p = self.Rho(p*Rho_0*c**2)/Rho_0*c**2
        A_phi = self.A(phi)
        a_phi = self.a+self.b*phi

        p_prime = -(rho_p+p)*(
            (4*pi*G*Rho_0/c**2*r**2*A_phi**4*p+mu)/(r*(1-2*mu))
            + r*omega**2/2 + a_phi*omega)

        # nu_prime = ((8*pi*G*rho_0*r**2*A_phi**4*p+2*mu)
        #             / (r*(1-2*mu))
        #             + r*omega**2)
        phi_prime = omega
        mu_prime = ((4*pi*G*Rho_0/c**2*r**2*A_phi**4*rho_p-mu)
                    / r
                    + r*(1-2*mu)*omega**2/2)
        omega_prime = ((4*pi*G*Rho_0/c**2*A_phi**4
                        / (1-2*mu))
                       * (a_phi*(rho_p-3*p)+r*omega*(rho_p-p))
                       - (2*(1-mu)*omega/(r*(1-2*mu))))
        print("p_prime\t",p_prime)
        print("mu_prime\t",mu_prime)
        return np.array([p_prime, phi_prime, omega_prime, mu_prime])

    def setup_initial(self, P_0, phi_0):
        self.Rho_0 = self.Rho(P_0)
        Rho_0 = self.Rho_0
        p_0 = P_0/Rho_0/c**2
        A_0 = self.A(phi_0)
        a_0 = self.a+self.b*phi_0
        print("p_0\t",p_0) 
        mu = (4*pi*G*Rho_0/c**2*A_0**4*self.r_start**2)/3
        p = p_0 + ((2*pi*G*Rho_0/c**2*A_0**4)/3
                   * (p_0+1)
                   * (a_0**2*(3*p_0-1)-(3*p_0+1))
                   * self.r_start**2)
        phi = phi_0 - ((2*pi*G*Rho_0/c**2*A_0**4)/3
                       * a_0*(3*p_0 - 1)
                       * self.r_start**2)
        omega = -2*((2*pi*G*Rho_0/c**2*A_0**4)/3
                    * a_0*(3*p_0 - 1)
                    * self.r_start)
        print("p\t",p)
        print("mu\t",mu)
        return np.array([p, phi, omega, mu])

    def match_external(self, r, x):
        p, phi, omega, mu = x

        # A_phi = self.A(phi)
        # a_phi = self.a+self.b*phi

        J = 2*(1-mu)+r**2*omega**2*(1-2*mu)
        K = 2*mu+r**2*omega**2*(1-2*mu)
        L = np.sqrt(4*mu**2
                    + 4*r**2*omega**2*(1-mu)*(1-2*mu)
                    + r**4*omega**4*(1-2*mu)**2)

        self.s = K/(2*np.sqrt(1-2*mu))*np.exp(-K/L*np.arctanh(L/J))
        self.a_A = (2*r*omega*(1-2*mu))/K
        self.phi_inf = phi + (2*r*omega*(1-2*mu)/L)*np.arctanh(L/J)
        self.M = r*self.s/G
        self.Q = self.a_A*self.M
        
    def integrate(self, P_0, phi_0):
        x = self.setup_initial(P_0, phi_0)
        O = scipy.integrate.ode(self.RHS)
        O.set_integrator('dopri5')
        O.set_initial_value(x, self.r_start)
        p = P_0/(self.Rho_0*c**2)
        rn = self.r_start
        rs = [rn]
        xs = [x]
        while p > 0 and O.successful():
            ro = rn
            xn = O.integrate(1e10, step=True)
            rn = O.t
            p, phi, omega, mu = xn
            rs.append(rn)
            xs.append(xn)
            print(rn, rn-ro)

        # now backtrack to find the zero between ro and rn

        # now use external matching to compute properties we 
        # actually care about

        return rs, xs
