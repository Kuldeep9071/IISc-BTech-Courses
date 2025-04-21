#include <bits/stdc++.h>
using namespace std;

// Part A

double f(double x) {
    return 0.2*x*exp(3.0*x) - 0.04*exp(3.0*x) + 0.04*exp(-2.0*x);
}
double f_prime(double x, double y) {
    return x*exp(3.0*x) - 2.0*y;
}
double f_double_prime(double x, double y) {
    return 3.0*x*exp(3.0*x) - 2.0*f_prime(x, y);
}

// Part B

double g(double x) {
    return x + 1.0/(1.0-x);
}
double g_prime(double x, double y) {
    return 1.0 + (x-y)*(x-y);
}
double g_double_prime(double x, double y) {
    return 2.0*(x-y)*g_prime(x, y);
}

// Part C

double p(double x) {
    return x*log(x) + 2.0*x;
}
double p_prime(double x, double y) {
    return 1.0 + y/x;
}
double p_double_prime(double x, double y) {
    return -y/(x*x) + p_prime(x, y);
}

// General Taylor Series Method for IVP
double Taylor_Series_Method_IVP(double x0, double y0, double x_end, double h, double (*f_prime)(double, double), double (*f_double_prime)(double, double)) {
    double x = x0;
    double y = y0;
    
    while(x < x_end) {
        double y1 = y + h*f_prime(x,y) + (h*h/2.0)*f_double_prime(x, y);
        y = y1;
        x += h;
        cout << "x: " << x << ", y: " << y << endl;
    }
    
    return y;
}

// Runge-Kutta Second Order Method
double Runge_Kutta_IVP_Order2(double x0, double y0, double h, double x_end, double (*f_prime)(double, double), double (*f)(double)) {
    double x = x0;
    double y = y0;
    
    while(x < x_end) {
        double k1 = h*f_prime(x, y);
        double k2 = h*f_prime(x + h/2.0, y + k1/2.0);
        y = y + k2; 
        x += h;
        cout << "x: " << x << ", y: " << y << endl;
    }
    
    return y;
}

// Trapezoidal Method
double Trapezoidal_IVP(double x0, double y0, double h, double x_end, double (*f_prime)(double, double), double (*f)(double)) {
    double x = x0;
    double y = y0;
    
    while (x < x_end) {
        double k1 = h * f_prime(x, y);
        double y1 = y + k1;
        double k2 = h * f_prime(x + h, y1);
        y = y + (k1 + k2) / 2.0;
        x += h;
        
        cout << "x: " << x << ", y: " << y << endl;
    }
    
    return y;
}

int main() {
    // Part A
    cout << "Part A:" << endl;

    cout << "Taylor Series Method: " << endl;
    double taylor_a = Taylor_Series_Method_IVP(0.0, 0.0, 1.0, 0.5, f_prime, f_double_prime);
    cout << "Final y at x = 1: " << taylor_a << endl;
    cout << "Actual Value: " << f(1.0) << endl;
    cout << endl;

    cout << "Runge-Kutta Second Order Method: " << endl;
    double rk_a = Runge_Kutta_IVP_Order2(0.0, 0.0, 0.5, 1.0, f_prime, f);
    cout << "Final y at x = 1: " << rk_a << endl;
    cout << "Actual Value: " << f(1.0) << endl;
    cout << endl;

    cout << "Trapezoidal Method: " << endl;
    double trap_a = Trapezoidal_IVP(0.0, 0.0, 0.5, 1.0, f_prime, f);
    cout << "Final y at x = 1: " << trap_a << endl;
    cout << "Actual Value: " << f(1.0) << endl;
    cout << endl;

    // Part B
    cout << "Part B:" << endl;

    cout << "Taylor Series Method: " << endl;
    double taylor_b = Taylor_Series_Method_IVP(2.0, 1.0, 3.0, 0.5, g_prime, g_double_prime);
    cout << "Final y at x = 3: " << taylor_b << endl;
    cout << "Actual Value: " << g(3.0) << endl;
    cout << endl;

    cout << "Runge-Kutta Second Order Method: " << endl;
    double rk_b = Runge_Kutta_IVP_Order2(2.0, 1.0, 0.5, 3.0, g_prime, g);
    cout << "Final y at x = 3: " << rk_b << endl;
    cout << "Actual Value: " << g(3.0) << endl;
    cout << endl;

    cout << "Trapezoidal Method: " << endl;
    double trap_b = Trapezoidal_IVP(2.0, 1.0, 0.5, 3.0, g_prime, g);
    cout << "Final y at x = 3: " << trap_b << endl;
    cout << "Actual Value: " << g(3.0) << endl;
    cout << endl;

    // Part C
    cout << "Part C:" << endl;

    cout << "Taylor Series Method: " << endl;
    double taylor_c = Taylor_Series_Method_IVP(1.0, 2.0, 2.0, 0.25, p_prime, p_double_prime);
    cout << "Final y at x = 2: " << taylor_c << endl;
    cout << "Actual Value: " << p(2.0) << endl;
    cout << endl;

    cout << "Runge-Kutta Second Order Method: " << endl;
    double rk_c = Runge_Kutta_IVP_Order2(1.0, 2.0, 0.25, 2.0, p_prime, p);
    cout << "Final y at x = 2: " << rk_c << endl;
    cout << "Actual Value: " << p(2.0) << endl;
    cout << endl;

    cout << "Trapezoidal Method: " << endl;
    double trap_c = Trapezoidal_IVP(1.0, 2.0, 0.25, 2.0, p_prime, p);
    cout << "Final y at x = 2: " << trap_c << endl;
    cout << "Actual Value: " << p(2.0) << endl;
    cout << endl;

    return 0;
}
