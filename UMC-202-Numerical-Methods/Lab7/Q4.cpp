#include <bits/stdc++.h>
using namespace std;

double f(double x) {
    return (x+1)*(x+1)-0.5*exp(x);
}

double f_prime(double x,double y) {
    return y-(x*x)+1;
}

double Runge_Kutta_Order_4(double x0, double y0, double h) {
    double x = x0, y = y0;

    double k1 = h * f_prime(x, y);
    double k2 = h * f_prime(x + h / 2, y + k1 / 2);
    double k3 = h * f_prime(x + h / 2, y + k2 / 2);
    double k4 = h * f_prime(x + h, y + k3);
    
    y += (k1 + 2 * k2 + 2 * k3 + k4) / 6;

    return y;
}

double Adams_4th_Orded_Predicator_Corrector(double x0,double y0,double h,double x){

    double x1 = x0;
    double y1 = y0;
    double x2 = x0+h;
    double x3 = x2+h;
    double x4 = x3+h;
    double y2 = Runge_Kutta_Order_4(x1,y1,h);
    double y3 = Runge_Kutta_Order_4(x2,y2,h);
    double y4 = Runge_Kutta_Order_4(x3,y3,h);
    

    for(double i = x4+h;i<=x;i+=h){
        double y5 = y4+h/24*(55*f_prime(x4,y4)-59*f_prime(x3,y3)+37*f_prime(x2,y2)-9*f_prime(x1,y1));
        y5 = y4+h/24*(9*f_prime(i,y5)+19*f_prime(x4,y4)-5*f_prime(x3,y3)+f_prime(x2,y2));
        x1 = x2;
        y1 = y2;
        x2 = x3;
        y2 = y3;
        x3 = x4;
        y3 = y4;
        x4 = i;
        y4 = y5;
    }
    return y4;
}

int main() {
    double x0 = 0, y0 = 0.5, h = 0.2, x = 2.0;
    cout << "y(2) = " << Adams_4th_Orded_Predicator_Corrector(x0, y0, h, x) << endl;
    cout<<"Actual value of y(2) = "<<f(x)<<endl;
    cout<<"Error = "<<abs(f(x)-Adams_4th_Orded_Predicator_Corrector(x0, y0, h, x))<<endl;

    return 0;
}