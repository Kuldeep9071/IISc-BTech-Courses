#include<bits/stdc++.h>
using namespace std;

vector<vector<double>> Difference(const vector<vector<double>> &data){
    vector<vector<double>> diff;
    diff.push_back(data[1]);
    for(int i=1;i<data[0].size();i++){
        vector<double> temp;
        for(int j=0;j<diff[i-1].size()-1;j++){
            temp.push_back(diff[i-1][j+1]-diff[i-1][j]);
        }
        diff.push_back(temp);
    }
    return diff;
}

double factorial(int n) {
    double ans=1.0;
    for (int i=2;i<=n;i++) ans*=i;
    return ans;
}

double C(double s, int k) {
    double ans=1.0;
    for (int i=0;i<k;i++) ans*=(s-i);
    return ans/factorial(k);
}


void Newton_Forward(vector<vector<double>> data,double x,int n){
    int max_degree=data[0].size()-1;
    if(n>max_degree){
        cout<<"Polynomial of degree "<<n<<"can't be interpolated with "<<max_degree+1<<"data points!\n";
        return;
    }
    vector<vector<double>> diff=Difference(data);
    double h=data[0][1]-data[0][0];
    double s=(x-data[0][0])/h;
    double ans=data[1][0];
    for(int k=1;k<=n;k++){
        ans+=C(s,k)*diff[k][0];
    }
    cout << "Interpolated value at x = " << x << " for degree " << n << ": " << ans << "\n";
}
int main(){
    cout<<"Part A\n";
    vector<vector<double>> data1 = {{-0.75, -0.5, -0.25, 0},{-0.07181250, 0.02475, 0.3349375, 1.101000}};
    Newton_Forward(data1,-0.3333,1);
    Newton_Forward(data1,-0.3333,2);
    Newton_Forward(data1,-0.3333,3);
    cout<<"\n Part B\n";
    vector<vector<double>> data2 = {{0.1, 0.2, 0.3, 0.4},{-0.62049958, -0.28398668, 0.00660095, 0.24842440}};
    Newton_Forward(data2,0.25,1);
    Newton_Forward(data2,0.25,2);
    Newton_Forward(data2,0.25,3);
    return 0;
}