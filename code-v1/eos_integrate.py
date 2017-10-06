from __future__ import division, print_function

import numpy as np
import scipy.integrate

pi = np.pi

G = 8
rho_0 = 8


class NeutronStar(object):

    def __init__(self, a, b, r_start=10.):
        self.a = a
        self.b = b
        self.r_start

    def A(self, phi):
        return np.exp(self.a*phi+self.b*phi**2/2)

    def rho(self, p):
        pass

    def RHS(self, r, x):
        """RHS for EOS integration

        Based on Horbatsch and Burgess 2011 eqns. 3.8--3.11
        """

        p, phi, omega, mu = x

        rho_p = self.rho(p)
        A_phi = self.A(phi)
        a_phi = self.a+self.b*phi

        p_prime = -(rho_p+p)*(
            (4*pi*G*rho_0*r**2*A_phi**4*p+mu)/(r*(1-2*mu))
            + r*omega**2/2 + a_phi*omega)

        # nu_prime = ((8*pi*G*rho_0*r**2*A_phi**4*p+2*mu)
        #             / (r*(1-2*mu))
        #             + r*omega**2)
        phi_prime = omega
        mu_prime = ((4*pi*G*rho_0*r**2*A_phi**4*rho_p-mu)
                    / r
                    + r*(1-2*mu)*omega**2/2)
        omega_prime = ((4*pi*G*rho_0*A_phi**4
                        / (1-2*mu))
                       * (a_phi*(rho_p-3*p)+r*omega*(rho_p-p))
                       - (2*(1-mu)*omega/(r*(1-2*mu))))

        return np.array([p_prime, phi_prime, omega_prime, mu_prime])

    def setup_initial(self, p_0, phi_0):
        A_0 = self.A(phi_0)
        a_0 = self.a+self.b*phi_0
        mu = (4*pi*G*rho_0*A_0**4*self.r_start**2)/3
        p = p_0 + ((2*pi*G*rho_0*A_0**4)/3
                   * (p_0+1)
                   * (a_0**2*(3*p_0-1)-(3*p_0+1))
                   * self.r_start**2)
        phi = phi_0 - ((2*pi*G*rho_0*A_0**4)/3
                       * a_0*(3*p_0 - 1)
                       * self.r_start**2)
        omega = -2*((2*pi*G*rho_0*A_0**4)/3
                       * a_0*(3*p_0 - 1)
                       * self.r_start)

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
        self.a_A = (2*R*omega*(1-2*mu))/K
        self.phi_inf = phi + (2*R*omega*(1-2*mu)/L)*np.arctanh(L/J)
        self.M = r*self.s/G
        self.Q = self.a_A*self.M
        
    def integrate(self, p_0, phi_0):
        x = self.setup_initial(p_0, phi_0)
        O = scipy.integrate.ode(self.RHS)
        O.set_integrator('dopri5')
        O.set_initial_value(x, self.r_start)
        p = p_0
        rn = self.r_start
        rs = [rn]
        xs = [x]
        while p > 0:
            ro = rn
            xn, rn = O.integrate(np.inf)
            p, phi, omega, mu = xn
            rs.append(rn)
            xs.append(xn)
        # now backtrack to find the zero between ro and rn

        # now use external matching to compute properties we 
        # actually care about

        return rs, xs
