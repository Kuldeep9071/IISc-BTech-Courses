#include<bits/stdc++.h>
using namespace std;

double f(double p){
    return (p*p*p)-(7.0*p*p)+(14.0*p)-6.0;
}
double bisection(double x,double y,int &ite){
    ite++;
    double m=x+(y-x)/2.0;
    if((y-x)/2.0<1e-4) return m;
    if(f(m)*f(x)<0)  return bisection(x,m,ite);
    else return bisection(m,y,ite);
}
int main(){
    int ite=0;
    double root1=bisection(0,1,ite);
    cout<<"Root 1:"<<root1<<",f(root1)="<<f(root1)<<"  and no. of iterations="<<ite<<"\n";
    ite=0;
    double root2=bisection(1,3.2,ite);
    cout<<"Root 2:"<<root2<<",f(root2)="<<f(root2)<<"  and no. of iterations="<<ite<<"\n";
    ite=0;
    double root3=bisection(3.2,4,ite);
    cout<<"Root 3:"<<root3<<",f(root3)="<<f(root3)<<"  and no. of iterations="<<ite<<"\n";
    return 0;
}