#include <bits/stdc++.h>
using namespace std;

double f(double x) {
    return log(x);
}

double f_triple_prime(double x) {
    return 2.0/(x*x*x);
}

double Three_Point_End_Point(double x, double h) {
    return (-3*f(x) + 4*f(x+h) - f(x+2*h))/(2*h);
}

double Three_Point_Mid_Point(double x, double h) {
    return (f(x+h) - f(x-h))/(2*h);
}

double Error_Three_Point_End_Point(double h) {
    return h*h/3.0 * f_triple_prime(1.8);
}

double Error_Three_Point_Mid_Point(double h) {
    return h*h/6.0 * f_triple_prime(1.8);
}

int main() {
    double h[] = {0.1, 0.05, 0.01};
    for (int i = 0; i < 3; i++) {
        cout << "For  h= " << h[i] << ", Derivative using 3 Point End Point= " << Three_Point_End_Point(1.8, h[i]) << " & Error: " << Error_Three_Point_End_Point(h[i]) << endl;
        cout << "For  h= " << h[i] << ", Derivative using 3 Point Mid Point= " << Three_Point_Mid_Point(1.8, h[i]) << " & Error: " << Error_Three_Point_Mid_Point(h[i]) << endl;
    }
    return 0;
}