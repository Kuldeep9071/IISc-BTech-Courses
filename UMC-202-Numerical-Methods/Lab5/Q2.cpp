#include <bits/stdc++.h>
using namespace std;

double f(double x) {
    return log(x);
}

double f_prime(double x) {
    return 1/x;
}

double backward_difference(double x, double h) {
    return (f(x) - f(x-h))/h;
}

double error(double h) {
    return h/2 * f_prime(1.8);
}

int main() {
    
    double h[] = {0.1, 0.05, 0.01};
    for (int i = 0; i < 3; i++) {
        cout << "For  h= " << h[i] << ", Derivative= " << backward_difference(1.8, h[i]) << " & Error: " << error(h[i]) << endl;
    }
    return 0;
}