#include <vector>
using namespace std; //needed?

//#define REALQUAD

#ifdef REALQUAD
extern "C" {
#include <quadmath.h>
}
#else
#include <float.h>
#define FLT128_MAX LDBL_MAX
#endif

#ifdef REALQUAD
typedef __float128 quad;
#else
typedef long double quad;
#endif
typedef vector<quad> vectq;

#ifdef REALQUAD
namespace std {
    quad pow(quad b, quad e) {
        return powq(b, e);
    };
    quad max(quad a, quad b) {
        return (a<b) ? b : a;
    };
    quad abs(quad a) {
        return (a<0) ? -a : a;
    };
};
#else
namespace std {
    quad max(quad a, quad b) {
        return (a<b) ? b : a;
    };
};
#endif

#include <boost/numeric/odeint.hpp>
using namespace boost::numeric::odeint;
#ifdef REALQUAD
// Cython can't cope with extern "C"
inline quad cpowq(quad b, quad e) {
    return powq(b,e);
}
inline quad cexpm1q(quad b) {
    return expm1q(b);
}
inline quad clog1pq(quad b) {
    return log1pq(b);
}
inline quad csqrtq(quad b) {
    return sqrtq(b);
}
#else
// Cython can't cope with extern "C"
inline quad cpowq(quad b, quad e) {
    return pow(b,e);
}
inline quad cexpm1q(quad b) {
    return expm1(b);
}
inline quad clog1pq(quad b) {
    return log1p(b);
}
inline quad csqrtq(quad b) {
    return sqrt(b);
}
#endif

// Argh! Cython won't let me do this for some reason
void vectq_set(vectq* v, size_t i, quad e) {
    (*v)[i] = e;
}

// Not sure I can make the right thing happen with references
// from Cython
typedef void (*callback_function)(
               vectq* x,
               vectq* dxdt,
               const quad t,
               void*arg);
class CRHS {
    callback_function cb;
    void* arg;
    long long evals;
    public:
        CRHS(callback_function cb, void*arg) {
            this->cb = cb;
            this->arg = arg;
            this->evals = 0;
        }
        void operator() ( const vectq &x,
                          vectq &dxdt, const quad t) {
            this->evals++;
            this->cb((vectq*)&x, &dxdt, t, this->arg);
        }
        void evaluate(const vectq *x,
                          vectq *dxdt, const quad t) {
            (*this)(*x, *dxdt, t);
        }
        long long n_evaluations() {
            return this->evals;
        }
};

// Apparently references do work!
void change_x(vector<quad> &x) {
    x[0] = -x[0];
}

struct iqobserver {
    vector<vectq> &m_r;
    iqobserver( vector<vectq> &r ) : m_r(r) {}
    void operator()(const vectq& x, quad t) {
        m_r.push_back(x);
    };
};

vector<vectq>* integrate_quad(const CRHS& RHS, 
        vectq initial_state, 
        const vectq &times,
        quad atol = 1e-10,
        quad rtol = 1e-10,
        quad initial_dt = 0) {
    typedef runge_kutta_fehlberg78<vectq,quad> error_stepper_type;
    //typedef bulirsch_stoer<vectq,quad> error_stepper_type;
    vector<vectq>* r = new vector<vectq>();

    iqobserver o = iqobserver(*r);

    if (initial_dt==0 && times.size()>1) {
        initial_dt = (times[1]-times[0])/100;
    }
    integrate_times(make_controlled<error_stepper_type>(atol, rtol),
            RHS, initial_state, times.begin(), times.end(), initial_dt,
            o);
    return r;
}


// typedef runge_kutta_fehlberg78<vectq,quad> error_stepper_type;
// typedef controlled_error_stepper<error_stepper_type> controlled_stepper_type;
// controlled_stepper_type make_stepper(quad rtol, quad atol) {
//     return make_controlled<error_stepper_type>(atol, rtol);
// };
typedef bulirsch_stoer<vectq, quad> controlled_stepper_type;
void integrate_to(CRHS&RHS,
        controlled_stepper_type &stepper,
        vectq&x, quad&t, quad&dt,
        quad t1) {
    while (t<t1) {
        // FIXME: make sure this works even with roundoff
        if (dt>t1-t) dt = t1-t; 
        stepper.try_step(RHS, x, t, dt);
    }
};


void integrate_to_with_delay(CRHS&RHS,
        bulirsch_stoer_dense_out<vectq, quad> &stepper,
        vectq &x,
        quad t) {
    while (stepper.current_time()<t) {
        stepper.do_step(RHS);
    }
    if (stepper.current_time()==t) {
        // This may be the first step; can't calculate state
        // Or we may have gotten lucky and don't need to bother interpolating
        x = stepper.current_state();
    } 
    if (stepper.previous_time()<=t) {
        stepper.calc_state(t, x);
    } else {
        // Trying to go the wrong way? Something is wrong.
    }
};


