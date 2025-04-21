#include<bits/stdc++.h>
using namespace std;

class RBTreeNode{
	public:
		int val=INT_MIN;
		RBTreeNode* parent=nullptr;
		RBTreeNode* left=nullptr;
		RBTreeNode* right=nullptr;
		char color;
		RBTreeNode(int val,char color='B'){
			this->val=val;
			this->color=color;
		}
};

