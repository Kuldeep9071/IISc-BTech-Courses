#include <bits/stdc++.h>
using namespace std;

void LagrangeInterpolation(vector<vector<double>> data,double x,int n){
    int max_degree=data.size()-1;
    if(n>max_degree){
        cout<<"Polynomial of degree "<<n<<" can't be interpolated with "<<max_degree+1<<" data points!\n";
        return;
    }
    double ans=0.0;
    for(int i=0;i<=n;i++){
        double temp=1.0;
        for(int j=0;j<=n;j++){
            if(i != j) temp*=(x-data[j][0])/(data[i][0]-data[j][0]);
        }
        ans+=temp*data[i][1];
    }
    cout << "Interpolated value at x = " << x << " for degree " << n << ": " << ans << "\n";
}

int main() {
    vector<vector<double>> data={{-0.75,-0.07181250},{-0.5,0.02475,},{-0.25,0.3349375},{0,1.101000}};
    cout<<"Part A\n";
    LagrangeInterpolation(data,-0.3333,1);
    LagrangeInterpolation(data,-0.3333,2);
    LagrangeInterpolation(data,-0.3333,3);
    cout<<"\n Part B\n";
    vector<vector<double>> data2={{0.1,-0.62049958},{0.2,-0.28398668},{0.3,0.00660095},{0.4,0.24842440}};
    LagrangeInterpolation(data2,0.25,1);
    LagrangeInterpolation(data2,0.25,2);
    LagrangeInterpolation(data2,0.25,3);
    return 0;
}