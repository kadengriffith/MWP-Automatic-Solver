class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

    def print_node(self):
        # Print held data
        print(self.data)

    def print_children(self):
        # Print inorder
        if self.left:
            self.left.print_node()

        print(self.data)

        if self.right:
            self.right.print_node()
