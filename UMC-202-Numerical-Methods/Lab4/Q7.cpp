#include <bits/stdc++.h>
using namespace std;

double f(double x, double y) {
    return log(x + 2 * y);
}

const double sqrt3_inv = 1.0 / sqrt(3.0);
const double gauss_points[2] = { -sqrt3_inv, sqrt3_inv }; 
const double gauss_weights[2] = { 1.0, 1.0 };             

double CompositeGaussianQuadrature(double a, double b, double c, double d, int n, int m) {
    double h=(b-a)/n, k=(d-c)/m, ans=0.0;
    
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++) {
            double x0=a+i*h;
            double x1=a+(i+1)*h;
            double y0=c+j*k;
            double y1=c+(j+1)*k;

            double hx=(x1-x0)/2.0;
            double cx=(x1+x0)/2.0;
            double hy=(y1-y0)/2.0; 
            double cy=(y1+y0)/2.0; 

        
            double sum=0.0;
            for (int p=0;p<2;p++) {     
                for (int q=0;q<2;q++){ 
                    double x_transformed=cx+ hx*gauss_points[p]; 
                    double y_transformed=cy+ hy*gauss_points[q]; 
                    sum+=gauss_weights[p]*gauss_weights[q]*f(x_transformed,y_transformed); 
                }
            }
            ans+=hx*hy*sum; 
        }
    }
    return ans;
}

int main() {
    double a=1.4,b=2.0,c=1.0,d=1.5;
    int n=1,m=1; 
    cout<<"CompositeGaussianQuadrature: "<<CompositeGaussianQuadrature(a,b,c,d,n,m)<<"\n";
    return 0;
}
