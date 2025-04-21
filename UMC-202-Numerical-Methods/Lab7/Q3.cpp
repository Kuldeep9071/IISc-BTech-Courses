#include <bits/stdc++.h>
using namespace std;

double f(double x){
    return (x+1)*(x+1)-0.5*exp(x);
}

double f_prime(double x,double y){
    return y-x*x+1;
}

double Adams_Bashforth_3_Step(double x0,double y0,double h,int n){
    double w0=y0, w1=x0+h*f_prime(x0,y0), w2=x0+h+ h*f_prime(x0+h,f_prime(x0+h,w1));
    for(int i=3;i<=n;i++){
        double w3=w2+h/24*(23*f_prime(w2,f(w2))-16*f_prime(w1,f(w1))+5*f_prime(w0,f(w0)));
        w0=w1;
        w1=w2;
        w2=w3;
    }
    return w1;
}

double Adams_Moulton_3_Step(double x0,double y0,double h,int n){
    double w0=y0,w1=x0+h*f_prime(x0,y0),w2=x0+h+ h*f_prime(x0+h,f_prime(x0+h,w1));
    for(int i=3;i<=n;i++){
        double w3=w2+h/24*(9*f_prime(w2,f(w2))+19*f_prime(w1,f(w1))-5*f_prime(w0,f(w0))+f_prime(w0-h,f(w0-h)));
        w0=w1;
        w1=w2;
        w2=w3;
    }
    return w1;
}

int main() {
    double x0=0,y0=0.5,h=0.1;
    int n=20;
    double yb=Adams_Bashforth_3_Step(x0,y0,h,n);
    cout<<"Adams Bashforth 3 Step: "<<yb<<endl;
    cout<<"Actual Value: "<<f(2)<<endl;
    cout<<"Error: "<<abs(yb-f(2))<<endl;
    cout<<endl;
    double ym=Adams_Moulton_3_Step(x0,y0,h,n);
    cout<<"Adams Moulton 3 Step: "<<ym<<endl;
    cout<<"Actual Value: "<<f(2)<<endl;
    cout<<"Error: "<<abs(ym-f(2))<<endl;

    return 0;
}