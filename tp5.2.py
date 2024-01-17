# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:46:35 2024

@author: julia
"""

#%% TASK 2 
#%%
#Importing necessary libraries
import networkx as nx
import matplotlib.pyplot as plt

# Definition of the Node class
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

#Definition of the BinarySearchTree class
class BinarySearchTree:
    def __init__(self):
        self.root = None

    #Method to insert a key into the tree
    def insert(self, key):
        self.root = self._insert(self.root, key)

    #Private method for recursive insertion
    def _insert(self, root, key):
        if root is None:
            return Node(key)
        if key < root.key:
            root.left = self._insert(root.left, key)
        elif key > root.key:
            root.right = self._insert(root.right, key)
        return root

    #Method to search for a key in the tree
    def search(self, key):
        return self._search(self.root, key)

    #Private method for recursive search
    def _search(self, root, key):
        if root is None:
            return False
        if root.key == key:
            return True
        if key < root.key:
            return self._search(root.left, key)
        return self._search(root.right, key)

    #Method to delete a key from the tree
    def delete(self, key):
        self.root = self._delete(self.root, key)

    #Private method for recursive deletion
    def _delete(self, root, key):
        if root is None:
            return root
        if key < root.key:
            root.left = self._delete(root.left, key)
        elif key > root.key:
            root.right = self._delete(root.right, key)
        else:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            root.key = self._get_min_value(root.right)
            root.right = self._delete(root.right, root.key)
        return root

    #Private method to get the minimum value
    def _get_min_value(self, root):
        while root.left is not None:
            root = root.left
        return root.key

    #Method for inorder traversal of the tree
    def inorder_traversal(self):
        result = []
        self._inorder_traversal(self.root, result)
        return result

    #Private method for recursive inorder traversal
    def _inorder_traversal(self, root, result):
        if root:
            self._inorder_traversal(root.left, result)
            result.append(root.key)
            self._inorder_traversal(root.right, result)

    #Method to graphically display the tree
    def plot_tree(self, title="Tree", color="skyblue"):
        G = nx.DiGraph()
        pos = self._build_graph(G, self.root)
        nx.draw(G, pos, with_labels=True, arrows=False, node_size=700, node_color=color, font_size=8)
        plt.title(title)
        plt.show()

    #Private method to recursively build the graph
    def _build_graph(self, G, node, pos=None, x=0, y=0, layer=1):
        if pos is None:
            pos = {node.key: (x, y)}
        else:
            pos[node.key] = (x, y)

        if node.left is not None:
            left_pos = (x - 1 / 2 ** layer, y - 1)
            G.add_edge(node.key, node.left.key)
            self._build_graph(G, node.left, pos, left_pos[0], left_pos[1], layer + 1)

        if node.right is not None:
            right_pos = (x + 1 / 2 ** layer, y - 1)
            G.add_edge(node.key, node.right.key)
            self._build_graph(G, node.right, pos, right_pos[0], right_pos[1], layer + 1)

        return pos

#Initial data
c = [49, 38, 65, 97, 64, 76, 13, 77, 5, 1, 55, 50, 24]

#Creating and populating the tree
bst_a = BinarySearchTree()
for num in c:
    bst_a.insert(num)

#Displaying the initial tree graphically
plt.figure(facecolor='pink')
bst_a.plot_tree(title="Tree C", color="pink")

#Adding the value 3
value_to_insert = 3
plt.figure(facecolor='lightblue')
bst_a.insert(value_to_insert)
#Displaying the tree after insertion
bst_a.plot_tree(title=f"Tree C after inserting value {value_to_insert}", color="lightblue")

#Deleting the value 13
value_to_delete = 13
plt.figure(facecolor='lightgreen')
bst_a.delete(value_to_delete)
# Displaying the tree after deletion
bst_a.plot_tree(title=f"Tree C after deleting value {value_to_delete}", color="lightgreen")

#Searching for the value 5
value_to_search = 5
result = bst_a.search(value_to_search)
print(f"Searching for value {value_to_search} in Tree A: {result}")
value_to_search = 52
result = bst_a.search(value_to_search)
print(f"Searching for value {value_to_search} in Tree A: {result}")

#Displaying the final tree graphically
plt.figure(facecolor='orange')
bst_a.plot_tree(title="Tree C final", color="orange")
plt.show()