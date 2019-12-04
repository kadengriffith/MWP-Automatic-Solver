from __future__ import absolute_import

import re
from classes.ExpressionTree import ExpressionTree
from classes.Stack import Stack

OPERATORS = set(['+', '-', '*', '/', '(', ')', '^'])
PRIORITY = {'+': 2, '-': 2, '*': 3, '/': 3, '^': 4}


class EquationConverter():
    def __init__(self, equation="DEFAULT"):
        self.original_equation = equation
        self.tree = ExpressionTree()
        self.equals_what = None

    def show_expression_tree(self):
        print(self.tree.levelorder())

    def expr_as_prefix(self):
        preorder_list = self.tree.preorder()
        prefix_expression = " ".join(preorder_list)

        return f"{self.equals_what} = {prefix_expression}"

    def expr_as_postfix(self):
        postorder_list = self.tree.postorder()
        postfix_expression = " ".join(postorder_list)

        return f"{self.equals_what} = {postfix_expression}"

    def expr_as_infix(self):
        inorder_list = self.tree.inorder()
        infix_expression = " ".join(inorder_list)

        return f"{self.equals_what} = {infix_expression}"

    def eqset(self, equation="DEFAULT"):
        self.original_equation = equation

        self.postfix_expression = self.__get_postfix_from_infix()

        self.__fill_tree()

    def is_eqset(self):
        return len(self.equation) > 0

    def __infix_to_postfix(self):
        filtered_expression, equation_equals = self.__filter_equation(
            self.original_equation)

        self.equals_what = equation_equals

        stack = Stack()
        output = ""

        split_expression = re.findall(r"(\d*\.?\d+|[^0-9])",
                                      filtered_expression)

        for char in split_expression:
            if char not in OPERATORS:
                output += char
            elif char == '(':
                stack.push(char)
            elif char == ')':
                while not stack.isEmpty() and stack.peek() != '(':
                    output += ' '
                    output += stack.pop()
                stack.pop()
            else:
                output += ' '

                while not stack.isEmpty() and stack.peek() != '(' and PRIORITY[char] <= PRIORITY[stack.peek()]:
                    output += stack.pop()

                stack.push(char)

        while not stack.isEmpty():
            output += ' '
            output += stack.pop()

        return output

    def __get_postfix_from_infix(self):
        return self.__infix_to_postfix()

    def __filter_equation(self, equation):
        equation_equals = ""

        # Clean the equation
        try:
            equation_equals = re.search(r"([a-z]+(\s+)?=|=(\s+)?[a-z]+)",
                                        equation).group(1)

            equation_equals = re.sub("=", "", equation_equals)
        except:
            pass

        equation = re.sub(r"([a-z]+(\s+)?=|=(\s+)?[a-z]+)",
                          "", equation)

        return equation.replace(' ', ""), equation_equals.replace(' ', "")

    def __fill_tree(self):
        # Start with the reversed postfix expression
        try:
            self.tree.tree_from_postfix(self.postfix_expression)
        except:
            pass


if __name__ == "__main__":
    import time

    print("Executing Equation Converter test...")

    start = time.time()

    converter = EquationConverter()

    print("\nTest 1:")
    eq = "x=9*(13-4)"
    converter.eqset(eq)
    converter.show_expression_tree()
    print("Infix with no parenthesis:", converter.expr_as_infix())
    print("Prefix:", converter.expr_as_prefix())
    print("Postfix:", converter.expr_as_postfix())

    print("\nTest 2:")
    eq = "x=9 * ((13-4) + 114)"
    converter.eqset(eq)
    converter.show_expression_tree()
    print("Infix with no parenthesis:", converter.expr_as_infix())
    print("Prefix:", converter.expr_as_prefix())
    print("Postfix:", converter.expr_as_postfix())

    print("\nTest 3:")
    eq = "(1.33333) * ((1399+4) / 114)=xyz"
    converter.eqset(eq)
    converter.show_expression_tree()
    print("Infix with no parenthesis:", converter.expr_as_infix())
    print("Prefix:", converter.expr_as_prefix())
    print("Postfix:", converter.expr_as_postfix())

    end = time.time()

    print("\n...done. Test took %.3fµs" % ((end - start) * 10000))
