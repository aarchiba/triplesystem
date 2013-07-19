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

// blerg. C++ type inference is *terrible*.
quad operator+(const quad&b, const quad&e) { return quad(b.v+e.v); }
quad operator+(const quad&b, const int&e) { return quad(b.v+e); }
quad operator+(const int&b, const quad&e) { return quad(b+e.v); }
quad operator+(const quad&b, const double&e) { return quad(b.v+e); }
quad operator+(const quad&b, const float&e) { return quad(b.v+e); }
quad operator+(const double&b, const quad&e) { return quad(b+e.v); }

quad operator-(const quad&b, const quad&e) { return quad(b.v-e.v); }
quad operator-(const int&b, const quad&e) { return quad(b-e.v); }
quad operator-(const quad&b, const int&e) { return quad(b.v-e); }
quad operator-(const quad&b, const float&e) { return quad(b.v-e); }

quad operator*(const quad&b, const quad&e) { return quad(b.v*e.v); }
quad operator*(const quad&b, const int&e) { return quad(b.v*e); }
quad operator*(const quad&b, const long int&e) { return quad(b.v*e); }
quad operator*(const int&b, const quad&e) { return quad(b*e.v); }
quad operator*(const long int&b, const quad&e) { return quad(b*e.v); }
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
    quad pow(quad b, int e) {
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
