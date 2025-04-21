#include <bits/stdc++.h>
using namespace std;

double f(double x){
    return (x+1)*(x+1)-0.5*exp(x);
}

double f_prime(double x,double y){
    return y-x*x+1;
}

double Adams_Bashforth_2_Step(double x0,double y0,double h,int n){
    double w0=y0,w1=x0+h*f_prime(x0,y0);
    for(int i=2;i<=n;i++){
        double w2=w1+h/2*(3*f_prime(w1,f(w1))-f_prime(w0,f(w0)));
        w0=w1;
        w1=w2;
    }
    return w1;
}
int main() {
    double x0=0,y0=0.5,h=0.1;
    int n=20;
    double y=Adams_Bashforth_2_Step(x0,y0,h,n);
    cout<<"Adams Bashforth 2 Step: "<<y<<endl;
    cout<<"Actual Value: "<<f(2)<<endl;
    cout<<"Error: "<<abs(y-f(2))<<endl;

    return 0;
}