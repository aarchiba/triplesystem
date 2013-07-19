#include <iostream>
#include <vector>
#include <float.h>
#include <limits>
#include <cmath>

using namespace std; //needed?

#include "quad_defs.hpp"

#include <Eigen/Dense>
#include "ppn.hpp"
using namespace Eigen;


typedef vector<long double> vectl;

#include <boost/numeric/odeint.hpp>
using namespace boost::numeric::odeint;

inline quad sqr(quad b) {
    return b*b;
}
inline long double sqr(long double b) {
    return b*b;
}

// Argh! Cython won't let me do this for some reason
void vectq_set(vectq* v, size_t i, quad e) {
    (*v)[i] = e;
}
void vectl_set(vectl* v, size_t i, long double e) {
    (*v)[i] = e;
}




// Useful code, finally

template<class num>
class CRHS {
        long long evals;
        void (*cb)(vector<num>*x, vector<num>*dxdt, const num t, void*arg);
        void* arg;
    public:
        CRHS(void (*cb)(vector<num>*x, vector<num>*dxdt, const num t, void*arg), 
                void*arg) : evals(0), cb(cb), arg(arg) { }
        void operator() ( const vector<num> &x,
                          vector<num> &dxdt, const num t) {
            this->evals++;
            this->cb((vector<num>*)&x, &dxdt, t, this->arg);
        }
        void evaluate(const vector<num> *x,
                          vector<num> *dxdt, const num t) {
            (*this)(*x, *dxdt, t);
        }
        long long n_evaluations() {
            return this->evals;
        }
};

class MisshapenState { };
#define G ((num)36779.59091405234)
#define c_ ((num)86400)
#define c2 ((num)7464960000)
template<class num>
void compute_ppn_rhs(const num x[], num dxdt[], const num gamma, const num beta,
		     const num Gamma01, const num Gamma02, const num Gamma12,
		     const num Theta01, const num Theta02, const num Theta12,
		     const num Gamma011, const num Gamma012, const num Gamma022,
		     const num Gamma100, const num Gamma102, const num Gamma122,
		     const num Gamma200, const num Gamma201, const num Gamma211,
		     const int matrix_mode, const num c_scale) {
  num x_scaled[21]; // unitless velocity, mass in lt-s
  num m0, m1, m2; // geometrized masses
  for(int k=0;k<3;k++) {
    for(int l=0;l<3;l++) {
      x_scaled[7*k+l] = x[7*k+l];
      x_scaled[7*k+l+3] = x[7*k+l+3]/c_/c_scale;
    }
    x_scaled[7*k+6] = x[7*k+6]*G/c2/c_scale/c_scale;
  }
  m0 = x_scaled[6];
  m1 = x_scaled[13];
  m2 = x_scaled[20];
  if (matrix_mode==0) {
    num a_direct[9];
    ppn_direct(x_scaled,a_direct,gamma,beta,
	       Gamma01*m0*m1,Gamma02*m0*m2,Gamma12*m1*m2,
	       Theta01*m0*m1,Theta02*m0*m2,Theta12*m1*m2,
	       Gamma011*m0*m1*m1, Gamma012*m0*m1*m2, Gamma022*m0*m2*m2,
	       Gamma100*m1*m0*m0, Gamma102*m1*m0*m2, Gamma122*m1*m2*m2,
	       Gamma200*m2*m0*m0, Gamma201*m2*m0*m1, Gamma211*m2*m1*m1);
    for(int j=0;j<3;j++) {
      for(int i=0;i<3;i++) {
	dxdt[j*7+i] = x[j*7+i+3];
	dxdt[j*7+i+3] = a_direct[j*3+i]*c2*c_scale*c_scale; // acceleration back into lt-s/day/day
      }
    }
  } else {
    Matrix< num, Dynamic, Dynamic > M(9,9);
    Matrix< num, Dynamic, 1 > b(9);
    if (matrix_mode>0) {
      ppn(x_scaled,M,b,gamma,beta,
	  Gamma01*m0*m1,Gamma02*m0*m2,Gamma12*m1*m2,
	  Theta01*m0*m1,Theta02*m0*m2,Theta12*m1*m2,
	  Gamma011*m0*m1*m1, Gamma012*m0*m1*m2, Gamma022*m0*m2*m2,
	  Gamma100*m1*m0*m0, Gamma102*m1*m0*m2, Gamma122*m1*m2*m2,
	  Gamma200*m2*m0*m0, Gamma201*m2*m0*m1, Gamma211*m2*m1*m1);
    } else {
      newton_lagrangian(x_scaled,M,b);
    }
    
    Matrix< num, Dynamic, 1 > a(9);
    if (matrix_mode==1 || matrix_mode<0) {
      a = M.ldlt().solve(b);
    } else if (matrix_mode==2) {
      a = M.fullPivHouseholderQr().solve(b);
    } else if (matrix_mode==3) {
      a = M.fullPivLu().solve(b);
    } 
    for(int j=0;j<3;j++) {
      for(int i=0;i<3;i++) {
	dxdt[j*7+i] = x[j*7+i+3];
	dxdt[j*7+i+3] = a(j*3+i)*c2*c_scale*c_scale; // acceleration back into lt-s/day/day
      }
    }
  }
}
template<class num>
class cKeplerRHS {
        long long &evals;
        const bool special, general;
        const num delta;
        const bool ppn_motion;
        const num gamma;
        const num beta;
        const num Gamma01, Gamma02, Gamma12;
        const num Theta01, Theta02, Theta12;
        const num Gamma011, Gamma012, Gamma022;
        const num Gamma100, Gamma102, Gamma122;
        const num Gamma200, Gamma201, Gamma211;
        const int matrix_mode;
        const num c_scale;
    public:
        cKeplerRHS(bool special, bool general, long long&evals, num delta,
		   bool ppn_motion, num gamma, num beta,
		   num Gamma01, num Gamma02, num Gamma12,
		   num Theta01, num Theta02, num Theta12,
 		   num Gamma011, num Gamma012, num Gamma022,
		   num Gamma100, num Gamma102, num Gamma122,
		   num Gamma200, num Gamma201, num Gamma211,
		   int matrix_mode, num c_scale) : 
	  evals(evals), special(special), general(general), delta(delta),
          ppn_motion(ppn_motion), gamma(gamma), beta(beta),
	  Gamma01(Gamma01), Gamma02(Gamma02), Gamma12(Gamma12),
          Theta01(Theta01), Theta02(Theta02), Theta12(Theta12),
	  Gamma011(Gamma011), Gamma012(Gamma012), Gamma022(Gamma022), 
	  Gamma100(Gamma100), Gamma102(Gamma102), Gamma122(Gamma122), 
	  Gamma200(Gamma200), Gamma201(Gamma201), Gamma211(Gamma211), 
	matrix_mode(matrix_mode), c_scale(c_scale) { };
        
        void kepler(const num x[], num dxdt[], const num t) {
            unsigned int i,j,k;
            num m_i, m_j;
            num r2_ij, cst;
	    
            for (i=0;i<21;i++) dxdt[i]=0;
	    if (ppn_motion) {
	      compute_ppn_rhs(x,dxdt,gamma,beta,
			      Gamma01,Gamma02,Gamma12,
			      Theta01,Theta02,Theta12,
 			      Gamma011, Gamma012, Gamma022,
			      Gamma100, Gamma102, Gamma122,
			      Gamma200, Gamma201, Gamma211,
			      matrix_mode, c_scale);
	    } else if (delta==0) {
	      for (i=0;i<3;i++) {
                for (k=0;k<3;k++)
		  dxdt[7*i+k] = x[7*i+k+3];
                m_i = x[7*i+6];
                for (j=0;j<i;j++) {
		  m_j = x[7*j+6];
		  r2_ij = 0;
		  for (k=0;k<3;k++)
		    r2_ij += sqr(x[7*j+k]-x[7*i+k]);
		  for (k=0;k<3;k++) {
		    cst = G*(x[7*j+k]-x[7*i+k])
		      *pow(r2_ij,-((num)3)/((num)2));
		    dxdt[7*i+k+3] +=  m_j*cst;
		    dxdt[7*j+k+3] += -m_i*cst;
		  }
                }
	      }
	    } else {
	      for (i=0;i<3;i++) {
                for (k=0;k<3;k++)
		  dxdt[7*i+k] = x[7*i+k+3];
                m_i = x[7*i+6]; // Inertial mass
                for (j=0;j<i;j++) {
		  m_j = x[7*j+6]; // Inertial mass
		  r2_ij = 0;
		  for (k=0;k<3;k++)
		    r2_ij += sqr(x[7*j+k]-x[7*i+k]);
		  for (k=0;k<3;k++) {
		    cst = G*(x[7*j+k]-x[7*i+k])
		      *pow(r2_ij,-((num)3)/((num)2));
		    if (j==0) cst *= (1+delta);
		    dxdt[7*i+k+3] +=  m_j*cst;
		    dxdt[7*j+k+3] += -m_i*cst;
		  }
                }
	      }
	    }
        }

        void relativity(const num x[], num dxdt[], const num t) {
            num slowing, temp, r, v2;
            unsigned int j,k;

            slowing = 0;
            if (this->special) {
                v2 = sqr(x[3])+sqr(x[4])+sqr(x[5]);
                slowing = expm1(-0.5*log1p(-v2/c2));
            }
            if (this->general) {
                for (j=1;j<3;j++) {
                    r = 0;
                    for (k=0;k<3;k++) 
                        r += sqr(x[7*j+k]-x[k]);
                    r = sqrt(r);
                    temp = expm1(-0.5*log1p(-2*G*x[7*j+6]/(r*c2)));
                    slowing = slowing + temp + slowing*temp;
                }
            }
            dxdt[21] = slowing;
        }
        void operator() ( const vector<num> &x,
                          vector<num> &dxdt, const num t) {
            unsigned int i,j,step,n;
            num x_a[22], dxdt_a[22];
            this->evals++;
            if (this->special || this->general) {
                step = 22;
            } else {
                step = 21;
            }
            n = x.size()/step;
            if (x.size() % step) throw MisshapenState();
            for (i=0; i<n; i++) {
                for (j=0; j<step; j++)
                    x_a[j] = x[j+i*step];
                this->kepler(x_a, dxdt_a, t);
                if (this->special || this->general)
                    this->relativity(x_a, dxdt_a, t);
                for (j=0; j<step; j++)
                    dxdt[j+i*step] = dxdt_a[j];
            }
        }
        void evaluate(const vector<num> *x,
                          vector<num> *dxdt, const num t) {
            (*this)(*x, *dxdt, t);
        }
};
template<class num>
num shapiro_delay(const vector<num>&x) { // in days
    const num c = 86400; // lt-s per day
    const num cst = -2*G/(c*c*c);
    num d = 0;
    for(int k=1;k<3;k++) {
        num delta_z, dr2=0;
        for (int i=0;i<3;i++) {
            delta_z = (x[i+7*k]-x[i]);
            dr2 += delta_z*delta_z;
        }
        d += cst*x[7*k+6]*log(delta_z+sqrt(dr2));
    }
    return d;
}

class WrongWay {
};

template<class num>
void do_step_dense(cKeplerRHS<num> rhs,
        bulirsch_stoer_dense_out<vector<num>, num> &stepper) {
    stepper.do_step(rhs);
};
template<class num>
void current_state(bulirsch_stoer_dense_out<vector<num>, num> &stepper,
        vector<num>&x) {
    x = stepper.current_state();
};
template<class num>
void previous_state(bulirsch_stoer_dense_out<vector<num>, num> &stepper,
        vector<num>&x) {
    x = stepper.previous_state();
};

template<class num, class System>
void integrate_to(System rhs,
        bulirsch_stoer<vector<num>, num> &stepper,
        vector<num> &x, 
        num &t, num &dt, num t1) {
    if (dt*(t1-t)<0) throw WrongWay();
    while (t<t1) {
        dt = min(dt,t1-t);
        stepper.try_step(rhs, x, t, dt);
    }
};

