#include <bits/stdc++.h>
using namespace std;

// Part A
double f(double x) {
    return x/(1+log(x));
}

double f_prime(double x, double y) {
    return y/x - y*y/(x*x);
}

double f_double_prime(double x, double y) {
    return -y/(x*x) + 2*y*y/(x*x*x) + f_prime(x, y)*(1/x - 2*y/x);
}

// Part B
double g(double x) {
    return 2*x/(1-2*x);
}

double g_prime(double x, double y) {
    return (y*y + y)/x;
}

double g_double_prime(double x, double y) {
    return -(y*y + y)/(x*x) + g_prime(x, y)*(2*y + 1)/x;
}

// Part C
double p(double x) {
    return sqrt(4 - 3*exp(-x*x));
}

double p_prime(double x, double y) {
    return -x*y + 4*x/y;
}

double p_double_prime(double x, double y) {
    return -y + 4/y + p_prime(x, y)*(-x - 4*x/(y*y));
}

// General Taylor Series Method for IVP
double Taylor_Series_Method_IVP(double x0, double y0, double x_end, double h,double (*f_prime)(double, double),double (*f_double_prime)(double, double)) {
    double x = x0;
    double y = y0;
    
    while(x < x_end) {
        double y1 = y + h*f_prime(x,y) + (h*h/2)*f_double_prime(x, y);
        y = y1;
        x += h;
        cout << "x: " << x << ", y: " << y << endl;
    }
    
    return y;
}

int main() {
    cout << "Part A: " << endl;
    Taylor_Series_Method_IVP(1, 1, 1.2, 0.1, f_prime, f_double_prime);
    cout << "Actual Value: " << f(1.2) << endl;
    cout << endl;
    
    cout << "Part B: " << endl;
    Taylor_Series_Method_IVP(1, -2, 3, 0.5, g_prime, g_double_prime);
    cout << "Actual Value: " << g(3) << endl;
    cout << endl;
    
    cout << "Part C: " << endl;
    Taylor_Series_Method_IVP(0, 1, 1, 0.25, p_prime, p_double_prime);
    cout << "Actual Value: " << p(1) << endl;
    
    return 0;
}
