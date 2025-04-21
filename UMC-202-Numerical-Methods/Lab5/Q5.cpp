#include <bits/stdc++.h>
using namespace std;

double f_prime(double x, double y) {
    return 1.0+(y/x);
}

double f_actual(double x) {
    return x*log(x)+2*x;
}

double Error(double t, double y, double h) {
    return abs(f_actual(t) - y);
}

double Euler_Method(double x0, double y0, double h, double xn) {
    double x = x0;
    double y = y0;
    cout<<x<<"\t "<<y<<"\n";
    while (x < xn) {
        y = y + h * f_prime(x, y);  
        x = x + h;
        cout << x << "\t " << y << "\n"; 
    }
    return y;
}

int main() {
    double x0 = 1.0;  
    double y0 = 2.0;  
    double h = 0.25;   
    double xn = 2.0;  
    double y=Euler_Method(x0, y0, h, xn);
    cout << "Approximate value of y(1) is " << y << "\n";
    cout << "Actual value of y(1) is " << f_actual(1) << "\n";
    cout << "Error is " << Error(2.0, y, h) << "\n";

    return 0;
}
