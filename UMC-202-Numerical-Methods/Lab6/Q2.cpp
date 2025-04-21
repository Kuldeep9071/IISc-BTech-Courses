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

// Runge-Kutta Second Order Method
double Runge_Kutta_IVP_Order_2(double x0, double y0, double h, double x_end, double (*f_prime)(double, double), double (*f)(double)) {
    double x = x0;
    double y = y0;
    while(x < x_end) {
        double k1 = h*f_prime(x, y);
        double k2 = h*f_prime(x + h/2, y + k1/2);
        y = y + k2; 
        x += h;
        cout<<"x: "<<x<<", y: "<<y<<endl;
    }
    return y;
}

int main() {
    cout << "Part A" << endl;
    double y_approx_A = Runge_Kutta_IVP_Order_2(1, 1, 0.1, 1.2, f_prime, f);
    cout << "Approximate y at x = 1.2: " << y_approx_A << endl;
    cout << "Actual: " << f(1.2) << endl;
    cout << endl;

    cout << "Part B" << endl;
    double y_approx_B = Runge_Kutta_IVP_Order_2(1, -2, 0.5, 3.0, g_prime, g);
    cout << "Approximate y at x = 3: " << y_approx_B << endl;
    cout << "Actual: " << g(3) << endl;
    cout << endl;

    cout << "Part C" << endl;
    double y_approx_C = Runge_Kutta_IVP_Order_2(0, 1, 0.25, 1.0, p_prime, p);
    cout << "Approximate y at x = 1: " << y_approx_C << endl;
    cout << "Actual: " << p(1) << endl;

    return 0;
}
