#include<bits/stdc++.h>
using namespace std;


// Time Complexity: O(n^3)
// Space Complexity: O(1)

int Number_of_TrianglesI(vector<int>& nums) {
    int n=nums.size(),ans=0;
    for(int i=0;i<n-2;i++){
        for(int j=i+1;j<n-1;j++){
            for(int k=j+1;k<n;k++){
                if(nums[i]+nums[j]>nums[k] && nums[j]+nums[k]>nums[i] && nums[k]+nums[i]>nums[j]) ans++;
            }
        }
    }
    return ans;
}


// Time Complexity: O(n^2 log(n))
// Space Complexity: O(1)

int BS(vector<int>& nums,int i,int j,int target){
    if(i>=j) return i;
    int m=i+(j-i)/2;
    if(nums[m]>=target) return min(m,BS(nums,i,m,target));
    return BS(nums,m+1,j,target);
}
int Number_of_TrianglesII(vector<int>& nums) {
    sort(nums.begin(),nums.end());
    int n=nums.size(),ans=0;
    for(int i=0;i<n-2;i++){
        for(int j=i+1;j<n-1;j++){
            int k;
            if(nums[n-1]<nums[i]+nums[j]) k=n;
            else k=BS(nums,j+1,n-1,nums[i]+nums[j]);
            ans+=(k-j-1);
        }
    }
    return ans;
}

// Time Complexity: O(n^2)
// Space Complexity: O(1)

int Number_of_TrianglesIII(vector<int>& nums) {
    sort(nums.begin(),nums.end());
    int n=nums.size(),ans=0;
    for (int i=n-1;i>=1;i--) {
        int l=0,r=i-1;
        while(l<r){
            if(nums[l]+nums[r]>nums[i]){
                ans+= r-l;
                r--;
            }
            else l++;
        }
    }
    return ans;
}



int main(){
	vector<int> arr={1,2,3,4,5,6,7,8,9,10};
	cout<<Number_of_TrianglesI(arr)<<"\n";
    cout<<Number_of_TrianglesII(arr)<<"\n";
    cout<<Number_of_TrianglesIII(arr)<<"\n";
    return 0;
}
