#include <iostream>
#include <vector>
#include <float.h>
#include <limits>
#include <cmath>

extern "C" {
#include <quadmath.h>
}

// for ppn.hpp, which needs to use __float128 to avoid template meltdowns
inline __float128 pow(__float128 b, int e) {
  return powq(b,e);
}
inline __float128 sqrt(__float128 b) {
  return sqrtq(b);
}

using namespace std; //needed?

//#include "quad_defs.hpp"

#include <Eigen/Dense>
#include "ppn.hpp"
using namespace Eigen;

#include <boost/multiprecision/float128.hpp>
typedef boost::multiprecision::float128 quad;
typedef vector<long double> vectl;
typedef vector<quad> vectq;

inline long double quad_to_longdouble(quad q) {
  return q.convert_to<long double>();
}

#include <boost/numeric/odeint.hpp>
using namespace boost::numeric::odeint;

inline quad log1p(quad q) {
  return log1pq(q.backend().value());
}
inline quad expm1(quad q) {
  return expm1q(q.backend().value());
}
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

// using boost's float128 causes GCC to melt down on these massive expressions
// strip down to raw __float128
void ppn(const quad xv[21], 
    quad M[9][9],
    quad b[9],
    const quad gamma, const quad beta,
    const quad Gamma01, const quad Gamma02, const quad Gamma12, 
    const quad Theta01, const quad Theta02, const quad Theta12,
    const quad Gamma011, const quad Gamma012, const quad Gamma022,
    const quad Gamma100, const quad Gamma102, const quad Gamma122,
    const quad Gamma200, const quad Gamma201, const quad Gamma211,
    const quad Omega, const quad Rc, const quad k2) {
  __float128 xvf[21];
  __float128 Mf[9][9];
  __float128 bf[9];
  for(int i=0;i<21;i++) {
    xvf[i] = xv[i].backend().value();
  }
  ppn(xvf, Mf, bf,
      gamma.backend().value(), beta.backend().value(),
      Gamma01.backend().value(), Gamma02.backend().value(), Gamma12.backend().value(), 
      Theta01.backend().value(), Theta02.backend().value(), Theta12.backend().value(),
      Gamma011.backend().value(), Gamma012.backend().value(), Gamma022.backend().value(),
      Gamma100.backend().value(), Gamma102.backend().value(), Gamma122.backend().value(),
      Gamma200.backend().value(), Gamma201.backend().value(), Gamma211.backend().value(),
      Omega.backend().value(), Rc.backend().value(), k2.backend().value());
  for(int i=0;i<9;i++) {
    b[i] = bf[i];
    for(int j=0;j<9;j++) {
      M[j][i] = Mf[j][i];
    }
  }
}
void ppn_direct(const quad xv[21], 
    quad a[9],
    const quad gamma, const quad beta,
    const quad Gamma01, const quad Gamma02, const quad Gamma12, 
    const quad Theta01, const quad Theta02, const quad Theta12,
    const quad Gamma011, const quad Gamma012, const quad Gamma022,
    const quad Gamma100, const quad Gamma102, const quad Gamma122,
    const quad Gamma200, const quad Gamma201, const quad Gamma211,
    const quad Omega, const quad Rc, const quad k2) {
  __float128 xvf[21];
  __float128 af[9];
  for(int i=0;i<21;i++) {
    xvf[i] = xv[i].backend().value();
  }
  ppn_direct(xvf, af,
	     gamma.backend().value(), beta.backend().value(),
	     Gamma01.backend().value(), Gamma02.backend().value(), Gamma12.backend().value(), 
	     Theta01.backend().value(), Theta02.backend().value(), Theta12.backend().value(),
	     Gamma011.backend().value(), Gamma012.backend().value(), Gamma022.backend().value(),
	     Gamma100.backend().value(), Gamma102.backend().value(), Gamma122.backend().value(),
	     Gamma200.backend().value(), Gamma201.backend().value(), Gamma211.backend().value(),
	     Omega.backend().value(), Rc.backend().value(), k2.backend().value());
  for(int i=0;i<9;i++) {
    a[i] = af[i];
  }
}

class MisshapenState { };
#define G_old ((num)36779.59091405234)
#define G ((num)36768.59290949113)
#define c_ ((num)86400)
#define c2 ((num)7464960000)
template<class num>
void compute_ppn_rhs(const num x[], num dxdt[], const num gamma, const num beta,
		     const num Gamma01, const num Gamma02, const num Gamma12,
		     const num Theta01, const num Theta02, const num Theta12,
		     const num Gamma011, const num Gamma012, const num Gamma022,
		     const num Gamma100, const num Gamma102, const num Gamma122,
		     const num Gamma200, const num Gamma201, const num Gamma211,
		     const num Omega, const num Rc, const num k2,
		     const int matrix_mode, const num c_scale) {
  num x_scaled[21]; // unitless velocity, mass in lt-s
  num m0, m1, m2; // geometrized masses
  num Omega_scaled; // rescale time
  for(int k=0;k<3;k++) {
    for(int l=0;l<3;l++) {
      x_scaled[7*k+l] = x[7*k+l];
      x_scaled[7*k+l+3] = x[7*k+l+3]/c_/c_scale;
    }
    x_scaled[7*k+6] = x[7*k+6]*G/c2/c_scale/c_scale;
  }
  Omega_scaled = Omega/c_/c_scale;
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
	       Gamma200*m2*m0*m0, Gamma201*m2*m0*m1, Gamma211*m2*m1*m1,
	       Omega_scaled, Rc, k2);
    for(int j=0;j<3;j++) {
      for(int i=0;i<3;i++) {
	dxdt[j*7+i] = x[j*7+i+3];
	dxdt[j*7+i+3] = a_direct[j*3+i]*c2*c_scale*c_scale; // acceleration back into lt-s/day/day
      }
    }
  } else {
    Matrix< num, Dynamic, Dynamic > Me(9,9);
    Matrix< num, Dynamic, 1 > be(9);
    num M[9][9];
    num b[9];
    if (matrix_mode>0) {
      ppn(x_scaled,M,b,gamma,beta,
	  Gamma01*m0*m1,Gamma02*m0*m2,Gamma12*m1*m2,
	  Theta01*m0*m1,Theta02*m0*m2,Theta12*m1*m2,
	  Gamma011*m0*m1*m1, Gamma012*m0*m1*m2, Gamma022*m0*m2*m2,
	  Gamma100*m1*m0*m0, Gamma102*m1*m0*m2, Gamma122*m1*m2*m2,
	  Gamma200*m2*m0*m0, Gamma201*m2*m0*m1, Gamma211*m2*m1*m1,
	  Omega, Rc, k2);
    } else {
      newton_lagrangian(x_scaled,M,b);
    }
    for(int j=0;j<9;j++) {
      be(j) = b[j];
      for(int i=0;i<9;i++) {
	Me(j,i) = M[j][i];
      }
    }
    Matrix< num, Dynamic, 1 > a(9);
    if (matrix_mode==1 || matrix_mode<0) {
      a = Me.ldlt().solve(be);
    } else if (matrix_mode==2) {
      a = Me.fullPivHouseholderQr().solve(be);
    } else if (matrix_mode==3) {
      a = Me.fullPivLu().solve(be);
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
    public:
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
        const num Omega, Rc, k2;
	const int matrix_mode;
        const num c_scale;
        const num pm_x;
        const num pm_y;
        const bool time_reciprocal;
        cKeplerRHS(bool special, bool general, long long&evals, num delta,
		   bool ppn_motion, num gamma, num beta,
		   num Gamma01, num Gamma02, num Gamma12,
		   num Theta01, num Theta02, num Theta12,
 		   num Gamma011, num Gamma012, num Gamma022,
		   num Gamma100, num Gamma102, num Gamma122,
		   num Gamma200, num Gamma201, num Gamma211,
		   const num Omega, const num Rc, const num k2,
		   int matrix_mode, num c_scale,
		   num pm_x, num pm_y, bool time_reciprocal) : 
	  evals(evals), special(special), general(general), delta(delta),
          ppn_motion(ppn_motion), gamma(gamma), beta(beta),
	  Gamma01(Gamma01), Gamma02(Gamma02), Gamma12(Gamma12),
          Theta01(Theta01), Theta02(Theta02), Theta12(Theta12),
	  Gamma011(Gamma011), Gamma012(Gamma012), Gamma022(Gamma022), 
	  Gamma100(Gamma100), Gamma102(Gamma102), Gamma122(Gamma122), 
	  Gamma200(Gamma200), Gamma201(Gamma201), Gamma211(Gamma211),
	  Omega(Omega), Rc(Rc), k2(k2),
	  matrix_mode(matrix_mode), c_scale(c_scale),
          pm_x(pm_x), pm_y(pm_y), time_reciprocal(time_reciprocal) { };
        
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
			      Omega, Rc, k2,
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
	    if (this->time_reciprocal) {
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
	    } else {
	      if (this->special) {
                v2 = sqr(x[3])+sqr(x[4])+sqr(x[5]);
                slowing = expm1(0.5*log1p(-v2/c2));
	      }
	      if (this->general) {
                for (j=1;j<3;j++) {
		  r = 0;
		  for (k=0;k<3;k++) 
		    r += sqr(x[7*j+k]-x[k]);
		  r = sqrt(r);
		  temp = expm1(0.5*log1p(-2*G*x[7*j+6]/(r*c2)));
		  slowing = slowing + temp + slowing*temp;
                }
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
num shapiro_delay(const vector<num>&x, const num gamma) { // in days
    const num c = 86400; // lt-s per day
    const num cst = -2*G*(1+gamma)/2/(c*c*c); 
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

