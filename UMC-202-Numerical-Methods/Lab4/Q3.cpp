#include<bits/stdc++.h>
using namespace std;

double f(double x){
    return pow(x,3)*exp(x);
}
double g(double x){
    return tan(x);
}
double CompositeTrapezoidalIntegrateI(double a,double b){
    double d=(b-a)/4.0,ans=0.0;
    while(a<b){
        ans+=(d*((f(a)+f(a+d))/2.0));
        a=a+d;
    }
    return ans;
}
double CompositeSimpsonIntegrateI(double a,double b){
    double d=(b-a)/4.0,ans=0.0;
    while(a<b){
        ans+=d/6.0*(f(a)+f(a+d)+4*(f(a+d/2.0)));
        a=a+d;
    }
    return ans;
}
double CompositeTrapezoidalIntegrateII(double a,double b){
    double d=(b-a)/8.0,ans=0.0;
    while(a<b){
        ans+=(d*((g(a)+g(a+d))/2.0));
        a=a+d;
    }
    return ans;
}
double CompositeSimpsonIntegrateII(double a,double b){
    double d=(b-a)/8.0,ans=0.0;
    while(a<b){
        ans+=d/6.0*(g(a)+g(a+d)+4*(g(a+d/2.0)));
        a=a+d;
    }
    return ans;
}
int main(){
    cout<<"Part-A\n";
    cout<<"Composite Trapezoidal: "<<CompositeTrapezoidalIntegrateI(-2.0,2.0)<<"\n";
    cout<<"Composite Simpson: "<<CompositeSimpsonIntegrateI(-2.0,2.0)<<"\n";
    cout<<"\nPart-B\n";
    cout<<"Composite Trapezoidal: "<<CompositeTrapezoidalIntegrateII(0,1.178097245)<<"\n";
    cout<<"Composite Simpson: "<<CompositeSimpsonIntegrateII(0,1.178097245)<<"\n";
}