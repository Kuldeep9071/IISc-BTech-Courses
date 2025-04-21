#include <bits/stdc++.h>
using namespace std;

double f1(double t,double w1,double w2) {
    return w2;
}

double f2(double t,double w1,double w2) {
    return 2.0*w2 - 2.0*w1 + exp(2.0*t)*sin(t);
}

double Runge_Kutta_Higher_Order_2(double a,double b,double n,double w1,double w2){
    double t=a,h=(b-a)/n;
    for(int i=1;i<=n;i++){
        double k11=h* f1(t,w1,w2);
        double k12=h* f2(t,w1,w2);
        double k21=h* f1(t+0.5*h,w1+k11/2,w2+k12/2);
        double k22=h* f2(t+0.5*h,w1+k11/2,w2+k12/2);
        double k31=h* f1(t+0.5*h,w1+k21/2,w2+k22/2);
        double k32=h* f2(t+0.5*h,w1+k21/2,w2+k22/2);
        double k41=h* f1(t+h,w1+k31,w2+k32);
        double k42=h* f2(t+h,w1+k31,w2+k32);

        w1+=(k11+2*k21+2*k31+k41)/6;
        w2+=(k12+2*k22+2*k32+k42)/6;
        t=a+i*h;
    }
    return w1;
}

int main() {
    double w1=-0.4,w2=-0.6;
    double a=0.0,b=1.0,h=0.1;
    int m=2,n=10;
    double R=Runge_Kutta_Higher_Order_2(a,b,n,w1,w2);
    cout<<"The value of y(1) is: "<<R<<endl;
    return 0;
}