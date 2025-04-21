#include<bits/stdc++.h>
using namespace std;

double f(double x){
    return pow(3.0*x*x+3.0,0.25);
}
double g(double x){
    return pow((pow(x,4)-3)/3.0,0.5);  
}
double h(double x){
    return 3*(pow(x,-1.0)+pow(x,-3.0));
}
double fixed_point(double p,int &ite){
    ite++;
    double b=f(p);
    if((b-p)<=1e-2)  return p;
    return fixed_point(b,ite);
}
int main(){
    int ite=0;
    cout<<"fixed point: "<<fixed_point(1.0,ite)<<" & no of iteration="<<ite<<"\n";
}