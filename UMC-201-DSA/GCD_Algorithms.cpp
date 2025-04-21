#include<bits/stdc++.h>
using namespace std;

// Brute Force Algorithm

int Bruteforce_gcd(int a, int b) {
    int gcd=1;
    int min_value=min(a, b);  

    for (int i=min_value;i>=1;i--) {
        if(a%i==0 && b%i==0){
            gcd=i;
            break;
        }
    }
    return gcd;
}

// Euclidean Algorithm - Recursive

int Eucleadian_gcd_recursive(int a,int b){
	if(a==0) return b;
	return gcd(b%a,a);
}

// Euclidean Algorithm - Iterative

int Eucleadian_gcd_iterative(int a,int b){
	while(a!=0){
		int temp=a;
		a=b%a;
		b=temp;
	}
	return b;
}

// Stein’s Algorithm or Binary GCD - Recursive

int Binary_gcd_recursive(int a,int b){
	if(a==0) return b;
	if(b==0) return a;
	int factor=1;
	while(!(a&1) && !(b&1)) b>>=1,a>>=1,factor<<=1;
	while(!(b&1))	b=b>>1;
	while(!(a&1))	a=a>>1;
	return factor*Binary_gcd_recursive(max(a,b)-min(a,b),min(a,b));
}

// Stein’s Algorithm or Binary GCD - Iterative

int Binary_gcd_iterative(int a,int b){
	if(a==0) return b;
	if(b==0) return a;
	int factor=1;
	while(!(a&1) && !(b&1)) b>>=1,a>>=1,factor<<=1;
	while(!(b&1))	b=b>>1;
	while(!(a&1))	a=a>>1;
	while(a!=b){
		if(a>b) a-=b;
		else b-=a;
	}
	return a*factor;
}

// C++ Inbuilt Function From GCC Compiler it may not work on NON GCC Compilers

int Inbuilt_gcd(int a,int b){
	return __gcd(a,b);
}

// C++ Inubilt Function From Standard Library from C++17 Compiler

int Standard_gcd(int a,int b){
	gcd(a,b);
}