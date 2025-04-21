#include <bits/stdc++.h>
using namespace std;

bool LagrangeInterpolation(vector<vector<double>> data, double x, int n, double expected) {
    int max_degree = data[0].size() - 1; 
    if (n>max_degree) {
        cout<<"Polynomial of degree "<<n<<" can't be interpolated with "<<max_degree+1<<" data points!\n";
        return false;
    }
    
    double ans=0.0; 
    for(int i=0;i<=n;i++){
        double temp=1.0; 
        for(int j=0;j<=n;j++){
            if(i!=j){
                temp*=(x-data[0][j])/(data[0][i]-data[0][j]);
            }
        }
        ans+=temp*data[1][i]; 
    }
    
    return abs(ans-expected)<1e-4;
}

int main() {
    vector<vector<double>> data={{-2,-1,0,1,2,3},{1,4,11,16,13,-4}};
   
    for (int i=1;i<=5;i++){
        bool flag=true;
        for(int j=i+1;j<=5;j++){
            if(!LagrangeInterpolation(data,data[0][j],i,data[1][j])){
                flag=false;
                break;
            }
        }
        if(flag){
            cout<<"Degree "<<i<<" polynomial correctly interpolates for all given data points\n";
        }
    }
    return 0;
}
