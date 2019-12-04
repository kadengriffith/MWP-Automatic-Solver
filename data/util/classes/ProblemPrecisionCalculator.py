class ProblemPrecisionCalculator():
    def __init__(self, reference, hypothesis):
        self.__r = reference
        self.__h = hypothesis

    def get_precision(self):
        # Uses a bias toward the real translation to
        #  determine a similarity score between space
        #  separated strings.
        r = self.__r.split()
        h = self.__h.split()

        comparison_list = []

        for i, char in enumerate(h):
            if i < len(r):
                if r[i] == char:
                    comparison_list.append(1)
                else:
                    comparison_list.append(0)
            else:
                comparison_list.append(0)

        precision = sum(comparison_list) / len(comparison_list)

        short_precision = "%1.4f" % precision

        return float(short_precision)


if __name__ == "__main__":
    pc = ProblemPrecisionCalculator("hello this is a test",
                                    "hi this is still a test")

    score = pc.get_precision()

    print(score)
