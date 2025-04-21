#include<bits/stdc++.h>
using namespace std;

double f(double x){
    return sqrt(1+x*x);
}
double RectangularIntegrate(double a,double b){
    return (b-a)*f(a);
}
double MidPointIntegrate(double a,double b){
    return ((b-a)/2.0)*(f(a)+f(b));
}

int main(){
    cout<<"Rectangular: "<<RectangularIntegrate(1,5)<<"\n";
    cout<<"Mid-Point: "<<MidPointIntegrate(1,5)<<"\n";
}