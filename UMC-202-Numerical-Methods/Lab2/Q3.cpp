#include<bits/stdc++.h>
using namespace std;
#include<math.h>

double F(double x){
    return exp(x)-x-1.0;
}
double FD(double x){
    return exp(x)-1.0;
}
double FDD(double x){
    return exp(x);
}
void Multiplicity(double x){
    if(F(x)!=0.0) cout<<"this is not a root of function: Multiplicity=0\n";
    else{
        if(FD(x)!=0.0) cout<<"Multiplicity:1\n";
        else if(FDD(x)!=0.0) cout<<"Multiplicity:2\n";
        else cout<<"Multiplicity is greater than 2\n";
    }
}
int main(){
    Multiplicity(0.0);
}