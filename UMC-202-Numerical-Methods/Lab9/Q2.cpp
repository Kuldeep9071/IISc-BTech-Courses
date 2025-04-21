#include <bits/stdc++.h>
using namespace std;

void Gauss_Jacobi(vector <vector <double>> A, vector <double> b, vector <double> x0,double tol) {
    int n = A.size();
    vector <double> x(n,0);
    double sum;
    int iter = 0;
    while (true) {
        for(int i=0;i<n;i++) {
            sum=0;
            for(int j=0;j<n;j++) {
                if (j!=i) {
                    sum += A[i][j]*x0[j];
                }
            }
            x[i]=(b[i]-sum)/A[i][i];
        }
        iter++;
        cout<<"(";
        for (int i=0;i<n-1;i++) {
            cout<<x[i]<<",";
        }
        cout<<x[n-1];
        cout<<")\n";
        bool flag = true;
        for (int i=0;i<n;i++) {
            if (abs(x[i]-x0[i])>tol) {
                flag = false;
                break;
            }
        }
        if (flag) break;
    
        x0 = x;
    }
    cout << "\nNumber of iterations: " << iter << endl;
}

int main() {
    vector <double> x0(4,0);
    vector <vector <double>> A = {{10,-1,2,0},{-1,11,-1,3},{2,-1,10,-1},{0,3,-1,8}};
    vector <double> b = {6,25,-11,15};
    double tol = 0.0001;

    Gauss_Jacobi(A,b,x0,tol);
    return 0;
}