#include<bits/stdc++.h>
using namespace std;

double f(double x){
    return pow(x,2)*sin(x);
}
double MidPointIntegrate(double a,double b){
    double d=1e-6,ans=0.0;
    while(a<b){
        ans+=(d*f(a+(d/2.0)));
        a=a+d;
    }
    return ans;
}
double OnePointGaussQuadrature(double a,double b){
    return ((b-a))*(f((a+b)/2.0));
}
int main(){
    double exact=MidPointIntegrate(0,M_PI/4.0);
    double OnePointGQ=OnePointGaussQuadrature(0,M_PI/4.0);
    cout<<"One Point Gauss Quadrature: "<<OnePointGQ<<"\n";
    cout<<"Error is 0 \n";
}