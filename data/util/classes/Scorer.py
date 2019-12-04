from __future__ import absolute_import

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from data.util.classes.ProblemPrecisionCalculator import ProblemPrecisionCalculator
import re


class Scorer():
    def __init__(self, hypothesis_actual_list):
        self.__hypothesis_list = hypothesis_actual_list

    def average(self, lst):
        return sum(lst) / len(lst)

    def get(self):
        bleu_average = []
        perfect = []
        precision = []

        for hypothesis, actual in self.__hypothesis_list:
            hypothesis = re.sub(r".0", "", hypothesis)
            actual = re.sub(r".0", "", actual)

            pc = ProblemPrecisionCalculator(actual, hypothesis)
            precision.append(pc.get_precision())

            # Format for BLEU
            bleu_hyp = hypothesis.split()
            bleu_act = actual.split()

            min_length = min(len(bleu_act), len(bleu_hyp))

            score = "%1.4f" % sentence_bleu([bleu_act],
                                            bleu_hyp,
                                            weights=(0.5, 0.5),
                                            smoothing_function=SmoothingFunction().method2)

            if score[0] == '1':
                perfect.append((hypothesis, actual))

            bleu_average.append(float(score))

        number_perfect = len(perfect)

        number_of_attempts = len(bleu_average)

        perfection_percentage = (number_perfect / number_of_attempts) * 100

        short_percentage = float("%3.2f" % perfection_percentage)

        avg_precision = (self.average(precision)) * 100

        short_precision = float("%3.2f" % avg_precision)

        bleu = self.average(bleu_average) * 100

        short_bleu = float("%3.2f" % (bleu))

        return number_of_attempts, short_percentage, short_precision, short_bleu
