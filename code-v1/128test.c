#include <quadmath.h>
#include <stdio.h>

int main(int argc, char*argv[]) {
    __float128 a, b;
    double c, d;
    long double e, f;
    int i = 0;
    a = 1;
    b = a;
    while (a+b>a) {
        b /= 2;
        i += 1;
    }
    printf("__float128\t2**(-%d) %Qg\n", i, b);
    i = 0;
    e = 1;
    f = e;
    while (e+f>e) {
        f /= 2;
        i += 1;
    }
    printf("long double\t2**(-%d) %Lg\n", i, f);
    i = 0;
    c = 1;
    d = c;
    while (c+d>c) {
        d /= 2;
        i += 1;
    }
    printf("double\t2**(-%d) %g\n", i, d);
    return 0;
}

