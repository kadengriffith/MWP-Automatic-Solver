import re


class NumberTag():
    def __init__(self, sentence, equation):
        self.__original_sentence = self.__ints(sentence)
        self.__original_equation = self.__ints(equation)

        self.__number_map = self.__map_numbers(self.__original_sentence,
                                               self.__original_equation)

        self.__tagged_sentence = self.__number_map[0]
        self.__tagged_equation = self.__number_map[1]
        self.__lookup_table = self.__number_map[2]

        self.__tags = ['x', 'y', 'z', 'j', 'q', 'v', 'w', 'r']

    def __map_numbers(self, sentence, equation):
        # Replaces numbers in a sentence with keyed tags
        splitput = sentence.split()
        spliteq = equation.split()
        lookup_dict = {}

        for i, word in enumerate(splitput):
            try:
                maybe_number = float(word)
                index = len(lookup_dict)

                key = f"<{self.__tags[index]}>"

                lookup_dict[key] = word

                splitput[i] = key
            except:
                pass

        adjust_dict = lookup_dict.copy()

        for i, word in enumerate(spliteq):
            try:
                for k, v in adjust_dict.items():
                    if word == v:
                        spliteq[i] = k
                        del adjust_dict[k]
                        break
            except:
                pass

        return " ".join(splitput), " ".join(spliteq), lookup_dict

    def __ints(self, sentence):
        # For example here, change 132.0 to 132, but leave 2.03 as is
        return re.sub(r"\.0[^0-9]", " ", sentence)

    def get_originals(self):
        return self.__original_sentence, self.__original_equation

    def get_masked(self):
        return self.__tagged_sentence, self.__tagged_equation, self.__lookup_table

    def apply_map(self, sentence, lookup):
        splitput = sentence.split()

        for i, word in enumerate(splitput):
            try:
                if word in lookup:
                    splitput[i] = lookup[word]
            except:
                pass

        return " ".join(splitput)


if __name__ == "__main__":
    problem, equation = "there are 128 books in a library . they are arranged on shelves that hold 4 books each . how many shelves are in the library ?", "x = 4 / 4"
    problem_tuple = NumberTag(problem, equation)

    print(problem_tuple.get_masked())

    print(problem_tuple.get_originals())
