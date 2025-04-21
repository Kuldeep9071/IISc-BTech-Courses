#include <bits/stdc++.h>
using namespace std;

/*
Time  Complexity: O(n^3)
Space Complexity: O(1)
*/
int Max_Subarray_Sum_Naive1(vector<int> &arr) {
    int n = arr.size();
    int maxSum = INT_MIN;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            int sum = 0;
            for (int k = i; k <= j; k++) {
                sum += arr[k];
            }
            maxSum = max(maxSum, sum);
        }
    }
    return maxSum;
}

/*
Time  Complexity: O(n^2)
Space Complexity: O(1)
*/

int Max_Subarray_Sum_Naive2(vector<int> &arr) {
    int n = arr.size();
    int maxSum = INT_MIN;
    for (int i = 0; i < n; i++) {
        int sum = 0;
        for (int j = i; j < n; j++) {
            sum += arr[j];
            maxSum = max(maxSum, sum);
        }
    }
    return maxSum;
}

/*
Time Complexity: O(nlogn)
Space Complexity: O(1)
*/

int Max_Subarray_Sum_Divide_and_Conquer(vector<int> &arr, int l, int r) {
    if (l == r) {
        return arr[l];
    }
    int m = (l + r) / 2;
    int leftMax = INT_MIN, rightMax = INT_MIN, sum = 0;
    for (int i = m; i >= l; i--) {
        sum += arr[i];
        leftMax = max(leftMax, sum);
    }
    sum = 0;
    for (int i = m + 1; i <= r; i++) {
        sum += arr[i];
        rightMax = max(rightMax, sum);
    }
    int maxLeft = Max_Subarray_Sum_Divide_and_Conquer(arr, l, m);
    int maxRight = Max_Subarray_Sum_Divide_and_Conquer(arr, m + 1, r);
    return max({maxLeft, maxRight, leftMax + rightMax});
}

/*
Time Complexity: O(n)
Space Complexity: O(1)
*/

vector<int> Max_Subarray_Sum_Divide_and_Conquer_Modified_Helper(vector<int> &arr, int l, int r) {
    if (l == r) {
        return {arr[l], arr[l], arr[l], arr[l]};
    }
    int m = (l + r) / 2;
    vector<int> left = Max_Subarray_Sum_Divide_and_Conquer_Modified_Helper(arr, l, m);
    vector<int> right = Max_Subarray_Sum_Divide_and_Conquer_Modified_Helper(arr, m + 1, r);
    int blockSum = left[3] + right[3];
    int prefixMax = max(left[0], left[3] + right[0]);
    int suffixMax = max(right[1], right[3] + left[1]);
    int maxSum = max({left[2], right[2], left[1] + right[0]});
    return {prefixMax, suffixMax, maxSum, blockSum};
} 

int Max_Subarray_Sum_Divide_and_Conquer_Modified(vector<int> &arr) {
    vector<int> res = Max_Subarray_Sum_Divide_and_Conquer_Modified_Helper(arr, 0, arr.size() - 1);
    return res[2];
}

/*
Time Complexity: O(n)
Space Complexity: O(1)
*/

int Kadanes_Algorithm(vector<int> &arr) {
    int n = arr.size();
    int maxSum = INT_MIN, sum = 0;
    for (int i = 0; i < n; i++) {
        sum = max(arr[i], sum + arr[i]);
        maxSum = max(maxSum, sum);
    }
    return maxSum;
}

// Main Function

int main() {
    vector<int> arr =  {-2, -3, 4, -1, -2, 1, 5, -3};
    cout << Max_Subarray_Sum_Naive1(arr) << endl;
    cout << Max_Subarray_Sum_Naive2(arr) << endl;
    cout << Max_Subarray_Sum_Divide_and_Conquer(arr, 0, arr.size() - 1) << endl;
    cout << Kadanes_Algorithm(arr) << endl;
    cout << Max_Subarray_Sum_Divide_and_Conquer_Modified(arr) << endl;
    return 0;
}