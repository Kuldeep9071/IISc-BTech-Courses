#include <bits/stdc++.h>
using namespace std;

/* Problem Statement:
    You are given an array(arr) Return Ture if there exits i!=j!=k such that arr[i]+arr[j]+arr[k]=0  else return false
*/



/* 
Time Complexity: O(n^3)
Space Complexity: O(1)
*/

bool Three_Sum_Naive(vector<int> &arr, int target) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            for (int k = j; k < n; k++) {
                if (arr[i] + arr[j] + arr[k] == target) {
                    return true;
                }
            }
        }
    }
    return false;
}

/*
Time Complexity: O(n^2 logn) 
Space Complexity: O(1)
*/

bool Binary_search(vector<int> &arr, int target) {
    int l = 0, r = arr.size() - 1;
    while (l <= r) {
        int m = (l + r) / 2;
        if (arr[m] == target) return true;
        else if (arr[m] < target) l = m + 1;
        else r = m - 1;
    }
    return false;
}

bool Three_Sum_Binary_Search(vector<int> &arr, int target) {
    int n = arr.size();
    sort(arr.begin(), arr.end());
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            int k = target - arr[i] - arr[j];
            if (Binary_search(arr, k)) {
                return true;
            }
        }
    }
    return false;
}

/*
Time Complexity: O(n^2)
Space Complexity: O(1)
*/

bool Three_Sum_Two_Pointers(vector<int> &arr, int target) {
    int n = arr.size();
    sort(arr.begin(), arr.end());
    for (int i = 0; i < n; i++) {
        int l = i + 1, r = n - 1;
        while (l < r) {
            int sum = arr[i] + arr[l] + arr[r];
            if (sum == target) return true;
            else if (sum < target) l++;
            else r--;
        }
    }
    return false;
}


int main() {

    vector<int> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int target = 29;

    if(Three_Sum_Naive(arr, target)) cout<<"True\n";
    else cout<<"False\n";
    if(Three_Sum_Binary_Search(arr, target)) cout<<"True\n";
    else cout<<"False\n";
    if(Three_Sum_Two_Pointers(arr, target)) cout<<"True\n";
    else cout<<"False\n";
    return 0;
}