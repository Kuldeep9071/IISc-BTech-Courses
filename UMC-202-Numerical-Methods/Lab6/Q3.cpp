#include <bits/stdc++.h>
using namespace std;

// Part A
double f_prime(double x, double y) {
    return y/x - y*y/(x*x);
}

double f(double x) {
    return x/(1+log(x));
}

// Part B
double g_prime(double x, double y) {
    return (y*y + y)/x;
}

double g(double x) {
    return 2*x/(1-2*x);
}

// Part C
double p_prime(double x, double y) {
    return -x*y + 4*x/y;
}

double p(double x) {
    return sqrt(4 - 3*exp(-x*x));
}

double Trapezoidal_IVP(double x0, double y0, double h, double x_end, double (*f_prime)(double, double), double (*f)(double)) {
    double x = x0;
    double y = y0;
    
    while (x < x_end) {
        double k1 = h * f_prime(x, y);
        double y1 = y + k1;
        double k2 = h * f_prime(x + h, y1);
        y = y + (k1 + k2) / 2;
        x += h;
        
        cout << "x: " << x << ", y: " << y << endl;
    }
    
    return y;
}

int main() {
    cout << "Part A\n" << endl;
    double y_approx_A = Trapezoidal_IVP(1, 1, 0.1, 1.2, f_prime, f);
    cout << "Approximate y at x = 1.2: " << y_approx_A << endl;
    cout << "Actual: " << f(1.2) << endl;
    cout << endl;

    cout << "Part B\n" << endl;
    double y_approx_B = Trapezoidal_IVP(1, -2, 0.5, 3.0, g_prime, g);
    cout << "Approximate y at x = 3: " << y_approx_B << endl;
    cout << "Actual: " << g(3) << endl;
    cout << endl;

    cout << "Part C\n" << endl;
    double y_approx_C = Trapezoidal_IVP(0, 1, 0.25, 1.0, p_prime, p);
    cout << "Approximate y at x = 1: " << y_approx_C << endl;
    cout << "Actual: " << p(1) << endl;

    return 0;
}
