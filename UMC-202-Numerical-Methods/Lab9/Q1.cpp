#include <bits/stdc++.h>
using namespace std;

// Gauss Elimination Back Substitution

vector<double> gaussElimination(vector<vector<double>> A, vector<double> b) {
    int n = b.size();
    for(int i=0; i<n; i++) {
        double maxEl = abs(A[i][i]);
        int maxRow = i;
        for(int k=i+1; k<n; k++) {
            if(abs(A[k][i]) > maxEl) {
                maxEl = abs(A[k][i]);
                maxRow = k;
            }
        }
        for(int k=i; k<n; k++) {
            double temp = A[maxRow][k];
            A[maxRow][k] = A[i][k];
            A[i][k] = temp;
        }
        double temp = b[maxRow];
        b[maxRow] = b[i];
        b[i] = temp;
        for(int k=i+1; k<n; k++) {
            double c = -A[k][i]/A[i][i];
            for(int j=i; j<n; j++) {
                if(i==j) {
                    A[k][j] = 0;
                } else {
                    A[k][j] += c*A[i][j];
                }
            }
            b[k] += c*b[i];
        }
    }
    vector<double> x(n);
    for(int i=n-1; i>=0; i--) {
        x[i] = b[i]/A[i][i];
        for(int k=i-1; k>=0; k--) {
            b[k] -= A[k][i]*x[i];
        }
    }
    return x;
}

int main() {
    double tol = 1e-2;
    vector<vector<double>> A = {{4,-1,1},{2,5,2},{1,2,4}};
    vector<double> b = {8,3,11};
    vector<double> x = gaussElimination(A,b);
    cout<<"x = "<<x[0]<<"\n";
    cout<<"y = "<<x[1]<<"\n";
    cout<<"z = "<<x[2]<<"\n";

    return 0;
}