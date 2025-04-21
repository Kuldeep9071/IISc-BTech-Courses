#include<bits/stdc++.h>
using namespace std;

double f(double x,double y){
    return log(x+2*y);
}

double CompositeSimpsonDoubleIntegral(double a, double b, double c, double d, int n, int m) {
    double h=(b-a)/n, k=(d-c)/m, ans=0.0;
    for(int i=0;i<n;i++) {
        for (int j=0;j<m;j++) {
            double x0=a+i*h;
            double x1=a+(i+1)*h;
            double y0=c+j*k;
            double y1=c+(j+1)*k;
            ans+=(f(x0,y0)+f(x1,y1)+f(x0,y1)+f(x1,y0)+ 
            4*f(x0+0.5*h,y0)+4*f(x0+0.5*h,y1)+ 
            4*f(x0,y0+0.5*k)+4*f(x1,y0+0.5*k)+ 
            16*f(x0+0.5*h,y0+0.5*k))*h*k/36;
        }
    }
    return ans;
}
int main(){
    cout<<"Composite Simpson Double Integral: "<<CompositeSimpsonDoubleIntegral(1.4,2.0,1.0,1.5,1.0,2.0)<<"\n";
}