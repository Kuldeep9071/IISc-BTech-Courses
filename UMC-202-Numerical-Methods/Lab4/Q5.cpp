#include<bits/stdc++.h>
using namespace std;

double f(double x){
    return sqrt(1+pow(x,2));
}
double MidPointIntegrate(double a,double b){
    double d=1e-6,ans=0.0;
    while(a<b){
        ans+=(d*f(a+(d/2.0)));
        a=a+d;
    }
    return ans;
}

double TwoPointGaussQuadrature(double a,double b){
    return ((b-a)/2.0)*(f((a+b-((b-a)/sqrt(3)))/2.0)+f((a+b+((b-a)/sqrt(3)))/2.0));
}

int main(){
    double exact=MidPointIntegrate(1.0,5.0);
    double TwoPointGQ=TwoPointGaussQuadrature(1.0,5.0);
    cout<<"Part-A\n";
    cout<<"Two Point Gauss Quadrature: "<<TwoPointGQ<<"\n";
    cout<<"Error : "<<((exact-TwoPointGQ)/exact)*100.0<<"\n";
}