#include "RBTree.h"

// Right Rotation about a node

void RightRotate_about(RBTreeNode* y){
	if(!y->left){
		cout<<"ERROR: Right Rotation Not Possible as left child of root is NULL Pointer!\n";
		return;
	}
	RBTreeNode* x=y->left;	
	RBTreeNode* b=x->right;
	RBTreeNode* temp=y->parent;
	if(temp){
		if(temp->left==y) temp->left=x;
		else temp->right=x;
	}
	y->left=b;
	if(b) b->parent=y;
	x->right=y;
	x->parent=temp;
	y->parent=x;
}

// Left Rotation about a node

void LeftRotate_about(RBTreeNode* y){
	if(!y->right){
		cout<<"ERROR: Left Rotation Not Possible as right child of root is NULL Pointer!\n";
		return;
	}
	RBTreeNode* x=y->right;
	RBTreeNode* b=x->left;
	RBTreeNode* temp=y->parent;
	if(temp){
		if(temp->left==y) temp->left=x;
		else temp->right=x;
	}
	y->right=b;
	if(b) b->parent=y;
	x->left=y;
	x->parent=temp;
	y->parent=x;
}

// Fixing the Red Black Tree after Insertion & Deletion

void Fix(RBTreeNode* node){
	if(!node->parent || node->parent->color=='B') return;
	RBTreeNode* grandPa=node->parent->parent; // GrandPa can't be NULL as parent is Red

	RBTreeNode* uncle;
	if(grandPa->left==node->parent)	uncle=grandPa->right;
	else uncle=grandPa->left;

	RBTreeNode* Father=node->parent;
	if(uncle && uncle->color=='R'){
		grandPa->color='R';
		if(uncle) uncle->color='B';
		Father->color='B';
		if(!grandPa->parent) grandPa->color='B'; // No More Fixing Required as it is root Just make it Black
		else Fix(grandPa);
	}
	else{
		grandPa->color='R';
		if(grandPa->left==Father && Father->left==node){
			RightRotate_about(grandPa);
			Father->color='B';
			return;			
		}
		else if(grandPa->right==Father && Father->right==node){
			LeftRotate_about(grandPa);
			Father->color='B';
			return;
		}
		else if(grandPa->left==Father && Father->right==node){
			LeftRotate_about(Father);
			RightRotate_about(grandPa);
			node->color='B';
			return;
		}
		else{
			RightRotate_about(Father);
			LeftRotate_about(grandPa);
			node->color='B';
			return;
		}
	}	
}
// Search in Red Black Tree - Normal BST Search

RBTreeNode* search(RBTreeNode* root,int val){
	if(!root) return nullptr;
	if(root->val==val) return root;
	if(root->val > val) return search(root->left,val);
	return search(root->right,val);
}

// Insertion in Red Black Tree

RBTreeNode* insertBST(RBTreeNode* root,int data,RBTreeNode* node){
	if(!root) return node;
	if(root->val > data){
		root->left=insertBST(root->left,data,node);
		root->left->parent=root;
	}
	else{
		root->right=insertBST(root->right,data,node);
		root->right->parent=root;
	}
	return root;
}

RBTreeNode* insert(RBTreeNode* root,int val){
	if(!root) return new RBTreeNode(val);
	RBTreeNode* node=new RBTreeNode(val,'R');
	insertBST(root,val,node);
	Fix(node);
	while(node->parent) node=node->parent; // Finding root after insertion
	return node;
}

// Deletion in Red Black Tree

void Delete(RBTreeNode* root,int val){
	RBTreeNode* node=search(root,val);
	if(!node){
		cout<<"ERROR: Node with value "<<val<<" Not Found in the Tree!\n";
		return;
	}
	if(!node->left && !node->right){
		if(node->parent){
			if(node->parent->left==node) node->parent->left=nullptr;
			else node->parent->right=nullptr;
		}
		delete node;
		return;
	}
	if(node->left && node->right){
		RBTreeNode* temp=node->right;
		while(temp->left) temp=temp->left;
		node->val=temp->val;
		Delete(node->right,temp->val);
		return;
	}
	RBTreeNode* child=node->left;
	if(!child) child=node->right;
	if(node->parent){
		if(node->parent->left==node) node->parent->left=child;
		else node->parent->right=child;
	}
	child->parent=node->parent;
	if(node->color=='B'){
		if(child->color=='R') child->color='B';
		else Fix(child);
	}
	delete node;
}

// Printing the Red Black Tree in Inorder Traversal (Sorted Order)

void printRBTree(RBTreeNode* root){
	if(root==nullptr) return;
	printRBTree(root->left);
	cout<<root->val<<"("<<root->color<<")   ";
	printRBTree(root->right);
}

int Height(RBTreeNode* root){
	if(!root) return 0;
	return 1+max(Height(root->left),Height(root->right));
}

// Black-Height of a Node in RB-Tree: Number of black nodes in the simple path from that node to the leaf node (NIL) excluding the node itself.

int BlackHeight(RBTreeNode* root){
	int Black_Height=0;
	while(root){
		if(root->color=='B') Black_Height++;
		root=root->left;
	}
	return Black_Height;
}


int main(){
	RBTreeNode* root=nullptr;
	root=insert(root,10);
	root=insert(root,20);
	root=insert(root,30);
	root=insert(root,15);
	// root=insert(root,25);
	// root=insert(root,35);
	// root=insert(root,40);
	// root=insert(root,0);
	// root=insert(root,5);
	// root=insert(root,3);
	cout<<"Height of Tree: "<<Height(root)<<"\n";
	cout<<"Black Height of Tree: "<<BlackHeight(root)<<"\n";
	printRBTree(root);
	Delete(root,10);
	cout<<"Height of Tree: "<<Height(root)<<"\n";
	cout<<"Black Height of Tree: "<<BlackHeight(root)<<"\n";
	printRBTree(root);
	cout<<root->val<<"("<<root->color<<")\n";
	cout<<"\n";
	return 0;
}
