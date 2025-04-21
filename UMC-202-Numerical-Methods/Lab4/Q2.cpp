#include<bits/stdc++.h>
using namespace std;

double f(double x){
    return 2/(x-4);
}
double g(double x){
    return exp(3*x)*sin(2*x);
}
double TrapezoidalIntegrateA(double a,double b){
    return (f(a)+f(b))*(b-a)/2.0;
}
double SimpsonIntegrateA(double a,double b){
    return (f(a)+4.0*f((a+b)/2.0)+f(b))*(b-a)/6.0;
}
double TrapezoidalIntegrateB(double a,double b){
    return (g(a)+g(b))*(b-a)/2.0;
}
double SimpsonIntegrateB(double a,double b){
    return (g(a)+4.0*g((a+b)/2.0)+g(b))*(b-a)/6.0;
}
int main(){
    cout<<"Part-A\n";
    cout<<"Trapazoidal: "<<TrapezoidalIntegrateA(0,0.5)<<"\n";
    cout<<"Simpson: "<<SimpsonIntegrateA(0,0.5)<<"\n";
    cout<<"\nPart-B\n";
    cout<<"Trapazoidal: "<<TrapezoidalIntegrateB(0,3.14159265358979323846/4.0)<<"\n";
    cout<<"Simpson: "<<SimpsonIntegrateB(0,3.14159265358979323846/4.0)<<"\n";
}