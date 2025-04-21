#include<bits/stdc++.h>
using namespace std;
#include<math.h>

double FD(double p){
    return 230.0*4.0*pow(p,3)+18.0*3.0*pow(p,2)+9.0*2*pow(p,1)-221.0;
}
double F(double p){
    return 230.0*pow(p,4)+18.0*pow(p,3)+9.0*pow(p,2)-221.0*p-9.0;
}

double Newtons_Method(double p,int &n){
    n++;
    double a=F(p);
    double b=FD(p);
    if(abs(a/b)<1e-5) return p;
    return Newtons_Method(p-a/b,n);
}

double Secant_Method(double p,double q,int &n){
    n++;
    double a=F(p);
    double b=F(q);
    double c=a*(p-q);
    double d=a-b;
    if(abs(1.0*c/d)<1e-6) return p;
    return Secant_Method(p-c/d,p,n);
}

double False_Position_Method(double a,double b,int &n){
    n++;
    double x=F(a),y=F(b);
    double p=(y*(b-a))/(y-x);
    double c=b-p;
    if(abs(p)<1e-6) return c;
    double m=F(c);
    if(x*m>0) a=c;
    else b=c;
    return False_Position_Method(a,b,n);
}

int main(){
    int n=0;
    cout<<"root in interval [0,1]:\n";
    cout<<"Newtons Method: "<<Newtons_Method(0.5,n)<<" and iter:"<<n<<"\n";
    n=0;
    cout<<"Secant Method: "<<Secant_Method(0.0,1.0,n)<<" and iter:"<<n<<"\n";
    n=0;
    cout<<"False Position Method: "<<False_Position_Method(0.0,1.0,n)<<" and iter:"<<n<<"\n";
    cout<<"\nroot in interval [-1,0]:\n";
    n=0;
    cout<<"Newtons Method: "<<Newtons_Method(-0.5,n)<<" and iter:"<<n<<"\n";
    n=0;
    cout<<"Secant Method: "<<Secant_Method(-1.0,0.0,n)<<" and iter:"<<n<<"\n";
    n=0;
    cout<<"False Position Method: "<<False_Position_Method(-1.0,0.0,n)<<" and iter:"<<n<<"\n";
}