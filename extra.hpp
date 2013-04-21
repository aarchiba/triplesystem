#include <iostream>
#include <vector>
#include <float.h>
#include <limits>
#include <cmath>
using namespace std; //needed?

extern "C" {
#include <quadmath.h>
}

#define FLT128_NAN (((__float128)0)/((__float128)0))
class quad { 
    // GCC has a bug where __float128 does not produce RTTI
    // This breaks the dense output integrators
    // So here's a thin wrapper
    public:
        __float128 v;
        quad() : v(FLT128_NAN) {};
        quad(__float128 val) : v(val) {};
        quad(const quad& val) : v(val.v) {};
        operator __float128() { return v; };
        const quad& operator +=(const quad&e) { v+=e.v; return *this; };
        const quad& operator -=(const quad&e) { v-=e.v; return *this; };
        const quad& operator *=(const quad&e) { v*=e.v; return *this; };
        const quad& operator /=(const quad&e) { v/=e.v; return *this; };
        //quad operator-() { return quad(-v); };
};
typedef vector<quad> vectq;
typedef vector<long double> vectl;

// blerg. C++ type inference is *terrible*.
quad operator+(const quad&b, const quad&e) { return quad(b.v+e.v); }
quad operator+(const quad&b, const int&e) { return quad(b.v+e); }
quad operator+(const int&b, const quad&e) { return quad(b+e.v); }
quad operator+(const quad&b, const double&e) { return quad(b.v+e); }
quad operator+(const double&b, const quad&e) { return quad(b+e.v); }

quad operator-(const quad&b, const quad&e) { return quad(b.v-e.v); }
quad operator-(const int&b, const quad&e) { return quad(b-e.v); }
quad operator-(const quad&b, const int&e) { return quad(b.v-e); }
quad operator-(const quad&b, const float&e) { return quad(b.v-e); }

quad operator*(const quad&b, const quad&e) { return quad(b.v*e.v); }
quad operator*(const quad&b, const int&e) { return quad(b.v*e); }
quad operator*(const int&b, const quad&e) { return quad(b*e.v); }
quad operator*(const unsigned int&b, const quad&e) { return quad(b*e.v); }
quad operator*(const double&b, const quad&e) { return quad(b*e.v); }

quad operator/(const quad&b, const quad&e) { return quad(b.v/e.v); }
quad operator/(const quad&b, const double&e) { return quad(b.v/e); }
quad operator/(const double&b, const quad&e) { return quad(b/e.v); }
quad operator/(const quad&b, const float&e) { return quad(b.v/e); }
quad operator/(const float&b, const quad&e) { return quad(b/e.v); }
quad operator/(const quad&b, const int&e) { return quad(b.v/e); }
quad operator/(const int&b, const quad&e) { return quad(b/e.v); }
quad operator/(const quad&b, const long unsigned int&e) { return quad(b.v/e); }
quad operator/(const long unsigned int&b, const quad&e) { return quad(b/e.v); }

quad operator-(const quad&b) { return quad(-b.v); }

quad operator<(const quad&b, const quad&e) { return b.v<e.v; }
quad operator>(const quad&b, const quad&e) { return b.v>e.v; }
quad operator>=(const quad&b, const quad&e) { return b.v>=e.v; }
quad operator<=(const quad&b, const quad&e) { return b.v<=e.v; }
quad operator<(const unsigned int&b, const quad&e) { return b<e.v; }
quad operator>(const unsigned int&b, const quad&e) { return b>e.v; }
quad operator>=(const unsigned int&b, const quad&e) { return b>=e.v; }
quad operator<=(const unsigned int&b, const quad&e) { return b<=e.v; }
quad operator<(const quad&b, const unsigned int&e) { return b.v<e; }
quad operator>(const quad&b, const unsigned int&e) { return b.v>e; }
quad operator>=(const quad&b, const unsigned int&e) { return b.v>=e; }
quad operator<=(const quad&b, const unsigned int&e) { return b.v<=e; }
quad operator<(const int&b, const quad&e) { return b<e.v; }
quad operator>(const int&b, const quad&e) { return b>e.v; }
quad operator>=(const int&b, const quad&e) { return b>=e.v; }
quad operator<=(const int&b, const quad&e) { return b<=e.v; }
quad operator<(const quad&b, const int&e) { return b.v<e; }
quad operator>(const quad&b, const int&e) { return b.v>e; }
quad operator>=(const quad&b, const int&e) { return b.v>=e; }
quad operator<=(const quad&b, const int&e) { return b.v<=e; }
quad operator<(const double&b, const quad&e) { return b<e.v; }
quad operator>(const double&b, const quad&e) { return b>e.v; }
quad operator>=(const double&b, const quad&e) { return b>=e.v; }
quad operator<=(const double&b, const quad&e) { return b<=e.v; }
quad operator<(const quad&b, const double&e) { return b.v<e; }
quad operator>(const quad&b, const double&e) { return b.v>e; }
quad operator>=(const quad&b, const double&e) { return b.v>=e; }
quad operator<=(const quad&b, const double&e) { return b.v<=e; }
quad operator<(const float&b, const quad&e) { return b<e.v; }
quad operator>(const float&b, const quad&e) { return b>e.v; }
quad operator>=(const float&b, const quad&e) { return b>=e.v; }
quad operator<=(const float&b, const quad&e) { return b<=e.v; }
quad operator<(const quad&b, const float&e) { return b.v<e; }
quad operator>(const quad&b, const float&e) { return b.v>e; }
quad operator>=(const quad&b, const float&e) { return b.v>=e; }
quad operator<=(const quad&b, const float&e) { return b.v<=e; }

quad operator==(const quad&b, const quad&e) { return b.v==e.v; }
quad operator==(const quad&b, const int&e) { return b.v==e; }
quad operator==(const int&b, const quad&e) { return b==e.v; }
quad operator==(const quad&b, const double&e) { return b.v==e; }
quad operator==(const double&b, const quad&e) { return b==e.v; }

namespace std {
    quad pow(quad b, quad e) {
        return powq(b, e);
    };
    quad max(quad a, quad b) {
        return (a<b) ? b : a;
    };
    quad fabs(quad a) {
        return (a<(quad)0) ? -a.v : a.v;
    };
    quad abs(quad a) {
        return (a<(quad)0) ? -a.v : a.v;
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
    inline long double expm1(long double b) {
        return expm1l(b);
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
    inline long double log1p(long double b) {
        return log1pl(b);
    }
    inline quad sqrt(quad b) {
        return sqrtq(b);
    }
    ostream& operator<<(ostream& ost, const quad&q) {
        return ost<<((long double)q.v);
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

class MisshapenState { };
#define G ((num)36779.59091405234)
#define c2 ((num)7464960000)
template<class num>
class cKeplerRHS {
        long long &evals;
        const bool special, general;
    public:
        cKeplerRHS(bool special, bool general, long long&evals) : 
            evals(evals), special(special), general(general) { };
        
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

