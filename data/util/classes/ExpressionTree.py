from __future__ import absolute_import

import re
from classes.Node import Node
from classes.Stack import Stack

# The Binary Tree Class
# WARNING: Supports only numbers and operators


class ExpressionTree():
    def __init__(self):
        self.root = None
        # Used for printing in various orders
        # This data is flushed after every traversal
        self.__ordered_data = []

    def insert(self, value, node="EMPTY_TREE"):
        # Inserts a node to the BinaryTree
        if node is "EMPTY_TREE":
            node = self.root

        if node is None:
            self.root = Node(value)

        else:
            if value < node.data:
                if node.left is None:
                    node.left = Node(value)
                else:
                    self.insert(value, node.left)
            else:
                if node.right is None:
                    node.right = Node(value)
                else:
                    self.insert(value, node.right)
        return 0

    def tree_from_postfix(self, postfix_expression):
        postfix_expression_array = re.findall(r"(\d*\.?\d+|[^0-9])",
                                              postfix_expression)

        elements = []
        for element in postfix_expression_array:
            if not re.match('\s+', element):
                elements.append(element)

        stack = Stack()

        for element in elements:
            if self.__is_operator(element) is True:
                subtree = Node(element)
                subtree.right = stack.pop()
                subtree.left = stack.pop()

                stack.push(subtree)

            else:
                stack.push(Node(element))

        self.root = stack.pop()

    def has_tree(self):
        return not self.root is None

    def inorder(self, node="DEFAULT"):
        # Returns the tree values in inorder
        node = self.__is_root_or_self(node)

        self.__traverse_inorder(node)

        return self.__get_tree()

    def __traverse_inorder(self, node):
        # Traverses the tree from the node provided in inorder
        if not node is None:
            self.__traverse_inorder(node.left)

            self.__ordered_data.append(node.data)

            self.__traverse_inorder(node.right)

    def preorder(self, node="DEFAULT"):
        # Returns the tree values in preorder
        node = self.__is_root_or_self(node)

        self.__traverse_preorder(node)

        return self.__get_tree()

    def __traverse_preorder(self, node):
        # Traverses the tree from the node provided in preorder
        if not node is None:
            self.__ordered_data.append(node.data)

            self.__traverse_preorder(node.left)

            self.__traverse_preorder(node.right)

    def postorder(self, node="DEFAULT"):
        # Returns the tree values in postorder
        node = self.__is_root_or_self(node)

        self.__traverse_postorder(node)

        return self.__get_tree()

    def __traverse_postorder(self, node):
        # Traverses the tree from the node provided in postorder
        if not node is None:
            self.__traverse_postorder(node.left)

            self.__traverse_postorder(node.right)

            self.__ordered_data.append(node.data)

    def levelorder(self, node="DEFAULT"):
        # Traverses the tree from the node provided in levelorder
        node = self.__is_root_or_self(node)

        if not node is None:
            result = []
            current = [node]

            while current:
                next_level = []
                values = []

                for tn in current:
                    values.append(tn.data)

                    if not tn.left is None:
                        next_level.append(tn.left)

                    if not tn.right is None:
                        next_level.append(tn.right)

                result.append(values)

                current = next_level

            return result

        else:
            return []

    def __is_root_or_self(self, node):
        # An easy way to simplify the use of the BinaryTree
        if node is "DEFAULT":
            return self.root
        else:
            return node

    def __flush_data(self):
        # Clear stored data
        self.__ordered_data = []

    def __is_operator(self, value):
        return value is '+' or value is '-' or value is '/' or value is '*' or value is '^'

    def __get_operator_precendence(self, operator):
        if operator is '^':
            return 4
        elif operator is '*' or operator is '/':
            return 3
        elif operator is '-' or operator is '+':
            return 2
        return 0

    def __get_tree(self):
        # Return the stored order of data ie. postorder
        # Erase the previous traversal method
        result = self.__ordered_data

        self.__flush_data()

        return result


if __name__ == "__main__":
    import time

    print("Executing Binary Tree test...")

    start = time.time()

    test_tree = BinaryTree()
    test_tree.insert(24)
    test_tree.insert(16)
    test_tree.insert(4)
    test_tree.insert(42)

    print(f"In order: {test_tree.inorder()}")
    print(f"Pre order: {test_tree.preorder()}")
    print(f"Post order: {test_tree.postorder()}")
    print(f"Level order: {test_tree.levelorder()}")

    end = time.time()

    print("...done. Test took %.3fµs." % ((end - start) * 1000000))
