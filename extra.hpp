#include <iostream>
#include <vector>
#include <float.h>
#include <limits>
using namespace std; //needed?

extern "C" {
#include <quadmath.h>
}

typedef __float128 quad;
typedef vector<quad> vectq;
typedef vector<long double> vectl;


namespace std {
    quad pow(quad b, quad e) {
        return powq(b, e);
    };
    quad max(quad a, quad b) {
        return (a<b) ? b : a;
    };
    quad fabs(quad a) {
        return (a<0) ? -a : a;
    };
    quad abs(quad a) {
        return (a<0) ? -a : a;
    };
    quad ceil(quad b) {
        return ceilq(b);
    }
    quad floor(quad b) {
        return floorq(b);
    }
    quad exp(quad b) {
        return expq(b);
    }
    long double max(long double a, long double b) {
        return (a<b) ? b : a;
    };
    inline quad expm1(quad b) {
        return expm1q(b);
    }
    inline quad sin(quad b) {
        return sinq(b);
    }
    inline quad cos(quad b) {
        return cosq(b);
    }
    inline quad log(quad b) {
        return logq(b);
    }
    inline quad log1p(quad b) {
        return log1pq(b);
    }
    inline quad sqrt(quad b) {
        return sqrtq(b);
    }
    ostream& operator<<(ostream& ost, const quad&q) {
        return ost<<((long double)q);
    }
    istream& operator>>(istream& ist, quad&q) {
        long double qq;
        ist>>qq;
        q = qq;
        return ist;
    }
};
namespace std {
  template<>
    struct numeric_limits<quad>
    {
      static _GLIBCXX_USE_CONSTEXPR bool is_specialized = true;

      static _GLIBCXX_CONSTEXPR quad 
      min() throw()  { return FLT128_MIN; }

      static _GLIBCXX_CONSTEXPR quad 
      max() throw() { return FLT128_MAX; }

      static _GLIBCXX_USE_CONSTEXPR int digits = FLT128_MANT_DIG;
      static _GLIBCXX_USE_CONSTEXPR int digits10 = FLT128_DIG;
      static _GLIBCXX_USE_CONSTEXPR bool is_signed = true;
      static _GLIBCXX_USE_CONSTEXPR bool is_integer = false;
      static _GLIBCXX_USE_CONSTEXPR bool is_exact = false;
      static _GLIBCXX_USE_CONSTEXPR int radix = __FLT_RADIX__;

      static _GLIBCXX_CONSTEXPR quad 
      epsilon() throw() { return FLT128_EPSILON; }

      static _GLIBCXX_CONSTEXPR quad 
      round_error() throw() { return 0.5; }

      static _GLIBCXX_USE_CONSTEXPR int min_exponent = FLT128_MIN_EXP;
      static _GLIBCXX_USE_CONSTEXPR int min_exponent10 = FLT128_MIN_10_EXP;
      static _GLIBCXX_USE_CONSTEXPR int max_exponent = FLT128_MAX_EXP;
      static _GLIBCXX_USE_CONSTEXPR int max_exponent10 = FLT128_MAX_10_EXP;

      static _GLIBCXX_USE_CONSTEXPR bool has_infinity = true;
      static _GLIBCXX_USE_CONSTEXPR bool has_quiet_NaN = true;
      static _GLIBCXX_USE_CONSTEXPR bool has_signaling_NaN = has_quiet_NaN;
      static _GLIBCXX_USE_CONSTEXPR float_denorm_style has_denorm
	= denorm_present;
      static _GLIBCXX_USE_CONSTEXPR bool has_denorm_loss 
        = true;

      static _GLIBCXX_CONSTEXPR quad 
      infinity() throw() { return __builtin_huge_val(); }

      static _GLIBCXX_CONSTEXPR quad 
      quiet_NaN() throw() { return __builtin_nan (""); }

      static _GLIBCXX_CONSTEXPR quad 
      signaling_NaN() throw() { return __builtin_nans (""); }

      static _GLIBCXX_CONSTEXPR quad 
      denorm_min() throw() { return FLT128_DENORM_MIN; }

      static _GLIBCXX_USE_CONSTEXPR bool is_iec559
	= has_infinity && has_quiet_NaN && has_denorm == denorm_present;
      static _GLIBCXX_USE_CONSTEXPR bool is_bounded = true;
      static _GLIBCXX_USE_CONSTEXPR bool is_modulo = false;

      static _GLIBCXX_USE_CONSTEXPR bool traps = false; // don't know
      static _GLIBCXX_USE_CONSTEXPR bool tinyness_before = false; // don't know
      static _GLIBCXX_USE_CONSTEXPR float_round_style round_style 
       = round_to_nearest;
    };
}

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

const quad G_mks = 6.67398e-11;
const quad c = 299792458;
const quad c2 = c*c;
const quad M_sun = 1.9891e30;
const quad G = G_mks / (c*c*c) * M_sun * 86400*86400;

template<class num>
class cKeplerRHS {
        long long evals;
        const bool special, general;
    public:
        cKeplerRHS(bool special, bool general) : 
            evals(0), special(special), general(general) { };
        
        void kepler(const num x[], num dxdt[], const num t) {
            unsigned int i,j,k;
            num m_i, m_j;
            num r2_ij, cst;

            for (i=0;i<21;i++) dxdt[i]=0;

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
            for (i=0; i<n; i++) {
                for (j=0; j<step; j++)
                    x_a[j] = x[j+i*step];
                this->kepler(x_a+i*step, dxdt_a+i*step, t);
                if (this->special || this->general)
                    this->relativity(x_a+i*step, dxdt_a+i*step, t);
                for (j=0; j<step; j++)
                    dxdt[j+i*step] = dxdt_a[j];
            }
        }
        void evaluate(const vector<num> *x,
                          vector<num> *dxdt, const num t) {
            (*this)(*x, *dxdt, t);
        }
        long long n_evaluations() {
            return this->evals;
        }
};

class WrongWay {
};

/* Problem: GCC doesn't provide typeinfo for __float128
template<class num>
void integrate_to_with_delay(CRHS<quad> rhs,
        bulirsch_stoer_dense_out<vector<quad>, quad> &stepper,
        vector<quad> &x,
        quad t) {
    while (stepper.current_time()<t) {
        stepper.do_step(rhs);
    }
    if (stepper.current_time()==t) {
        // This may be the first step; can't calculate state
        // Or we may have gotten lucky and don't need to bother interpolating
        x = stepper.current_state();
    } else if (stepper.previous_time()<=t) {
        stepper.calc_state(t, x);
    } else {
        // Trying to go the wrong way? Something is wrong.
        // How do C++ exceptions work?
        // How does Cython respond to C++ exceptions?
        //throw WrongWay();
    }
};
*/
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

