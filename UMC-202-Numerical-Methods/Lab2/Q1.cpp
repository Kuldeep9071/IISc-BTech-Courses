#include<bits/stdc++.h>
using namespace std;
#include<math.h>
//1.a
double FD_1(double p){
    return 3*p*p-4*p; // 1.a
}
double F_1(double p){
    return p*p*p-2*p*p-5.0;
}

double Newtons_Method_1(double p,int &n){
    n++;
    double a=F_1(p);
    double b=FD_1(p);
    if(abs(a/b)<1e-5) return p;
    return Newtons_Method_1(p-a/b,n);
}
// 1.b
double FD_2(double p){
    return 2*p-2*exp(-2*p)-2*(1.0-p)*exp(-p);
}
double F_2(double p){
    return p*p-2.0*p*exp(-p)+exp(-2*p);
}

double Newtons_Method_2(double p,int &n){
    n++;
    double a=F_2(p);
    double b=FD_2(p);
    if(abs(a/b)<1e-5) return p;
    return Newtons_Method_2(p-a/b,n);
}
// 1.c
double FD_3(double p){
    return pow(8,-p)*(pow(4,p)*pow(p,2)*(3*pow(2,p)+log(8))+3*pow(2,p)-3*pow(2,p)*(pow(2,p+1.0)+log(4))*p+log(8));
}
double F_3(double p){
    return pow(p,3)-3*pow(p,2)*pow(2,-p)+3*p*pow(4,-p)-pow(8,-p);
}

double Newtons_Method_3(double p,int &n){
    n++;
    double a=F_3(p);
    double b=FD_3(p);
    if(abs(a/b)<1e-5) return p;
    return Newtons_Method_3(p-a/b,n);
}

int main(){
    int n=0;
    cout<<Newtons_Method_1(2.5,n)<<" and Number of iterations: "<<n<<"\n";
    n=0;
    cout<<Newtons_Method_2(0.5,n)<<" and Number of iterations: "<<n<<"\n";
    n=0;
    cout<<Newtons_Method_3(0.64,n)<<" and Number of iterations: "<<n<<"\n";
}