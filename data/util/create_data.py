from __future__ import absolute_import

import os
import sys
import yaml
import json
import pickle
import re
import random
import time
from word2number import w2n

from classes.EquationConverter import EquationConverter
from utils import to_binary

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

USE_GENERATED = False

try:
    TEST_SPLIT = int(sys.argv[2])
except:
    TEST_SPLIT = 0.05

# i.e. "plus" instead of '+'
WORDS_FOR_OPERATORS = False

# Composite list of MWPs
PROBLEM_LIST = []

# The same list with all equations converted from infix to cleaned infix
CLEAN_INFIX_CONVERTED_PROBLEM_LIST = []

# The same list with all equations converted from infix to Polish notation
POLISH_CONVERTED_PROBLEM_LIST = []

# The same list with all equations converted from infix to Reverse Polish notation
REVERSE_POLISH_CONVERTED_PROBLEM_LIST = []

# The generated data (not used in testing)
GENERATED = []

# Dataset specific
AI2 = []
ILLINOIS = []
COMMONCORE = []
MAWPS = []

KEEP_INFIX_PARENTHESIS = True
MAKE_IND_SETS = True

# Large test sets
PREFIX_TEST = []
POSTFIX_TEST = []
INFIX_TEST = []

# The file containing the set info
DATA_STATS = os.path.join(DIR_PATH,
                          "../DATA.md")

with open(os.path.join(DIR_PATH, f"../../{sys.argv[1]}"), 'r', encoding='utf-8-sig') as yaml_file:
    settings = yaml.load(yaml_file, Loader=yaml.FullLoader)

# Random seed for shuffling the data
SEED = settings["seed"]
random.seed(SEED)


def one_sentence_clean(text):
    # Clean up the data and separate everything by spaces
    text = re.sub(r"(?<!Mr|Mr|Dr|Ms)(?<!Mrs)(?<![0-9])(\s+)?\.(\s+)?", " . ",
                  text, flags=re.IGNORECASE)
    text = re.sub(r"(\s+)?\?(\s+)?", " ? ", text)
    text = re.sub(r",", "", text)
    text = re.sub(r"^\s+", "", text)
    text = text.replace('\n', ' ')
    text = text.replace("'", " '")
    text = text.replace('%', ' percent')
    text = text.replace('$', ' $ ')
    text = re.sub(r"\.\s+", " . ", text)
    text = re.sub(r"\s+", ' ', text)

    sent = []
    for word in text.split(' '):
        try:
            sent.append(str(w2n.word_to_num(word)))
        except:
            sent.append(word)

    return ' '.join(sent)


def formatNumber(num):
    if float(num) % 1 == 0:
        return str(int(num))
    else:
        return str(num)


def remove_point_zero(text):
    return text
    # temp = []
    # for word in text.split(' '):
    #     try:
    #         temp.append(formatNumber(word))
    #     except:
    #         temp.append(word)
    # sentence = ' '.join(temp)

    # return re.sub(r"\.0([^0-9])?", ' ', sentence)


def to_lower_case(text):
    # Convert strings to lowercase
    try:
        return text.lower()
    except:
        return text


def remove_variables(lst):
    new_lst = []
    for problem in lst:
        new_problem = []
        for elements in problem:
            if elements[0] == "equation":
                try:
                    text = re.sub(r"([a-z]+(\s+)?=|=(\s+)?[a-z]+)",
                                  "", elements[1])
                    text = re.sub(r"^\s+", "", text)
                    new_problem.append((elements[0], text))
                except:
                    pass
            else:
                new_problem.append((elements[0], elements[1]))
        new_lst.append(new_problem)

    return new_lst


def word_operators(text):
    if WORDS_FOR_OPERATORS:
        rtext = re.sub(r"\+", "add", text)
        rtext = re.sub(r"(-|\-)", "subtract", rtext)
        rtext = re.sub(r"\/", "divide", rtext)
        rtext = re.sub(r"\*", "multiply", rtext)
        return rtext
    return text


def transform_AI2():
    print("\nWorking on AI2 data...")

    problem_list = []

    with open(os.path.join(DIR_PATH, "../datasets/AI2/questions.txt"), "r") as fh:
        content = fh.readlines()

    iterator = iter(content)

    for i in range(len(content)):
        if i % 3 == 0 or i == 0:
            # The MWP
            question_text = one_sentence_clean(content[i].strip())
            question_text = remove_point_zero(question_text)

            eq = remove_point_zero(content[i + 2].strip())

            problem = [("question", to_lower_case(question_text)),
                       ("equation", to_lower_case(eq)),
                       ("answer", content[i + 1].strip())]

            if problem != []:
                problem_list.append(problem)
                AI2.append(problem)

            # Skip to the next MWP in data
            next(iterator)
            next(iterator)

    total_problems = int(len(content) / 3)

    print(f"-> Retrieved {len(problem_list)} / {total_problems} problems.")

    print("...done.\n")

    return "AI2"


def transform_CommonCore():
    print("\nWorking on CommonCore data...")

    problem_list = []

    with open(os.path.join(DIR_PATH, "../datasets/CommonCore/questions.json"), encoding='utf-8-sig') as fh:
        json_data = json.load(fh)

    for i in range(len(json_data)):
        # A MWP
        problem = []

        has_all_data = True

        data = json_data[i]
        if "sQuestion" in data and "lEquations" in data and "lSolutions" in data:
            for key, value in data.items():
                if key == "sQuestion" or key == "lEquations" or key == "lSolutions":
                    if len(value) == 0 or (len(value) > 1 and (key == "lEquations" or key == "lSolutions")):
                        has_all_data = False

                    if key == "sQuestion":
                        desired_key = "question"

                        value = one_sentence_clean(value)
                        value = remove_point_zero(value)

                        problem.append((desired_key,
                                        to_lower_case(value)))
                    elif key == "lEquations":
                        desired_key = "equation"

                        value = value[0]

                        value = remove_point_zero(value)

                        problem.append((desired_key,
                                        to_lower_case(value)))
                    elif key == "lSolutions":
                        desired_key = "answer"

                        problem.append((desired_key,
                                        to_lower_case(value[0])))
                    else:
                        problem.append((desired_key,
                                        to_lower_case(value)))

        if has_all_data == True and problem != []:
            problem_list.append(problem)
            COMMONCORE.append(problem)

    print(f"-> Retrieved {len(problem_list)} / {len(json_data)} problems.")

    print("...done.\n")

    return "CommonCore"


def transform_Illinois():
    print("\nWorking on Illinois data...")

    problem_list = []

    with open(os.path.join(DIR_PATH, "../datasets/Illinois/questions.json"), encoding='utf-8-sig') as fh:
        json_data = json.load(fh)

    for i in range(len(json_data)):
        # A MWP
        problem = []

        has_all_data = True

        data = json_data[i]
        if "sQuestion" in data and "lEquations" in data and "lSolutions" in data:
            for key, value in data.items():
                if key == "sQuestion" or key == "lEquations" or key == "lSolutions":
                    if len(value) == 0 or (len(value) > 1 and (key == "lEquations" or key == "lSolutions")):
                        has_all_data = False

                    if key == "sQuestion":
                        desired_key = "question"

                        value = one_sentence_clean(value)
                        value = remove_point_zero(value)

                        problem.append((desired_key,
                                        to_lower_case(value)))
                    elif key == "lEquations":
                        desired_key = "equation"

                        value = value[0]

                        value = remove_point_zero(value)

                        problem.append((desired_key,
                                        to_lower_case(value)))
                    elif key == "lSolutions":
                        desired_key = "answer"

                        problem.append((desired_key,
                                        to_lower_case(value[0])))
                    else:
                        problem.append((desired_key,
                                        to_lower_case(value)))

        if has_all_data == True and problem != []:
            problem_list.append(problem)
            ILLINOIS.append(problem)

    print(f"-> Retrieved {len(problem_list)} / {len(json_data)} problems.")

    print("...done.\n")

    return "Illinois"


def transform_MaWPS():
    print("\nWorking on MaWPS data...")

    path = os.path.join(DIR_PATH, "../datasets/MaWPS/questions.json")

    problem_list = []

    with open(path, encoding='utf-8-sig') as fh:
        json_data = json.load(fh)

    for i in range(len(json_data)):
            # A MWP
        problem = []

        has_all_data = True

        data = json_data[i]
        if "sQuestion" in data and "lEquations" in data and "lSolutions" in data:
            for key, value in data.items():
                if key == "sQuestion" or key == "lEquations" or key == "lSolutions":
                    if len(value) == 0 or (len(value) > 1 and (key == "lEquations" or key == "lSolutions")):
                        has_all_data = False

                    if key == "sQuestion":
                        desired_key = "question"

                        value = one_sentence_clean(value)
                        value = remove_point_zero(value)

                        problem.append((desired_key,
                                        to_lower_case(value)))
                    elif key == "lEquations":
                        if len(value) > 1:
                            continue

                        desired_key = "equation"

                        value = value[0]

                        value = remove_point_zero(value)

                        problem.append((desired_key,
                                        to_lower_case(value)))
                    elif key == "lSolutions":
                        desired_key = "answer"

                        problem.append((desired_key,
                                        to_lower_case(value[0])))
                    else:
                        problem.append((desired_key,
                                        to_lower_case(value)))

        if has_all_data == True and problem != []:
            problem_list.append(problem)
            MAWPS.append(problem)

    print(f"-> Retrieved {len(problem_list)} / {len(json_data)} problems.")

    print("...done.\n")

    return "MaWPS"


def transform_custom():
    print("\nWorking on generated data...")

    path = os.path.join(DIR_PATH, "../gen.p")

    problem_list = []

    with open(path, "rb") as fh:
        file_data = pickle.load(fh)

        for problem in file_data:
            if problem != []:
                problem_list.append(problem)
                GENERATED.append(problem)

    print(f"-> Retrieved {len(problem_list)} / {len(file_data)} problems.")

    print("...done.\n")

    return "Custom"


def transform_all_datasets():
    total_datasets = []

    # Iteratively rework all the data
    total_datasets.append(transform_AI2())
    total_datasets.append(transform_CommonCore())
    total_datasets.append(transform_Illinois())
    total_datasets.append(transform_MaWPS())
    if USE_GENERATED:
        total_datasets.append(transform_custom())

    return total_datasets


def convert_to(l, t):
    output = []

    for p in l:
        p_dict = dict(p)

        ol = []

        discard = False

        for k, v in p_dict.items():
            if k == "equation":
                convert = EquationConverter()
                convert.eqset(v)

                if t == "infix":
                    ov = convert.expr_as_infix()
                elif t == "prefix":
                    ov = convert.expr_as_prefix()
                elif t == "postfix":
                    ov = convert.expr_as_postfix()

                if re.match(r"[a-z] = .*\d+.*", ov):
                    ol.append((k, word_operators(ov)))
                else:
                    discard = True
            else:
                ol.append((k, v))

        if not discard:
            output.append(ol)

    return output


if __name__ == "__main__":
    print("Transforming all original datasets...")
    print(f"Splitting {(1 - TEST_SPLIT) * 100}% for training.")
    print("NOTE: Find resulting data binaries in the data folder.")

    total_filtered_datasets = transform_all_datasets()

    # Split
    AI2_TEST = AI2[:int(len(AI2) * TEST_SPLIT)]
    AI2 = AI2[int(len(AI2) * TEST_SPLIT):]

    COMMONCORE_TEST = COMMONCORE[:int(len(COMMONCORE) * TEST_SPLIT)]
    COMMONCORE = COMMONCORE[int(len(COMMONCORE) * TEST_SPLIT):]

    ILLINOIS_TEST = ILLINOIS[:int(len(ILLINOIS) * TEST_SPLIT)]
    ILLINOIS = ILLINOIS[int(len(ILLINOIS) * TEST_SPLIT):]

    MAWPS_TEST = MAWPS[:int(len(MAWPS) * TEST_SPLIT)]
    MAWPS = MAWPS[int(len(MAWPS) * TEST_SPLIT):]

    if USE_GENERATED:
        GENERATED_TEST = GENERATED[:int(len(GENERATED) * TEST_SPLIT)]
        GENERATED = GENERATED[int(len(GENERATED) * TEST_SPLIT):]
        random.shuffle(GENERATED)

    random.shuffle(AI2)
    random.shuffle(COMMONCORE)
    random.shuffle(ILLINOIS)
    random.shuffle(MAWPS)

    PROBLEM_LIST = AI2 + COMMONCORE + ILLINOIS + MAWPS + GENERATED

    # Randomize
    random.shuffle(PROBLEM_LIST)

    # AI2 testing data
    test_pre_ai2 = convert_to(AI2_TEST, "prefix")
    test_pos_ai2 = convert_to(AI2_TEST, "postfix")
    if KEEP_INFIX_PARENTHESIS:
        test_inf_ai2 = remove_variables(AI2_TEST)
        test_inf_ai2 = test_inf_ai2[:len(test_pos_ai2)]
    else:
        test_inf_ai2 = convert_to(AI2_TEST, "infix")

    to_binary(os.path.join(DIR_PATH, "../test_ai_prefix.p"),
              test_pre_ai2)
    to_binary(os.path.join(DIR_PATH, "../test_ai_postfix.p"),
              test_pos_ai2)
    to_binary(os.path.join(DIR_PATH, "../test_ai_infix.p"),
              test_inf_ai2)

    # AI2 training data
    pre_ai2 = convert_to(AI2, "prefix")
    pos_ai2 = convert_to(AI2, "postfix")
    if KEEP_INFIX_PARENTHESIS:
        inf_ai2 = remove_variables(AI2)
        inf_ai2 = inf_ai2[:len(pos_ai2)]
    else:
        inf_ai2 = convert_to(AI2, "infix")

    if MAKE_IND_SETS:
        to_binary(os.path.join(DIR_PATH, "../train_ai_prefix.p"),
                  pre_ai2)
        to_binary(os.path.join(DIR_PATH, "../train_ai_postfix.p"),
                  pos_ai2)
        to_binary(os.path.join(DIR_PATH, "../train_ai_infix.p"),
                  inf_ai2)

    # Common Core testing data
    test_pre_common = convert_to(COMMONCORE_TEST, "prefix")
    test_pos_common = convert_to(COMMONCORE_TEST, "postfix")
    if KEEP_INFIX_PARENTHESIS:
        test_inf_common = remove_variables(COMMONCORE_TEST)
        test_inf_common = test_inf_common[:len(test_pos_common)]
    else:
        test_inf_common = convert_to(COMMONCORE_TEST, "infix")

    to_binary(os.path.join(DIR_PATH, "../test_cc_prefix.p"),
              test_pre_common)
    to_binary(os.path.join(DIR_PATH, "../test_cc_postfix.p"),
              test_pos_common)
    to_binary(os.path.join(DIR_PATH, "../test_cc_infix.p"),
              test_inf_common)

    # Common Core training data
    pre_common = convert_to(COMMONCORE, "prefix")
    pos_common = convert_to(COMMONCORE, "postfix")
    if KEEP_INFIX_PARENTHESIS:
        inf_common = remove_variables(COMMONCORE)
        inf_common = inf_common[:len(pos_common)]
    else:
        inf_common = convert_to(COMMONCORE, "infix")

    if MAKE_IND_SETS:
        to_binary(os.path.join(DIR_PATH, "../train_cc_prefix.p"),
                  pre_common)
        to_binary(os.path.join(DIR_PATH, "../train_cc_postfix.p"),
                  pos_common)
        to_binary(os.path.join(DIR_PATH, "../train_cc_infix.p"),
                  inf_common)

    # Illinois testing data
    test_pre_il = convert_to(ILLINOIS_TEST, "prefix")
    test_pos_il = convert_to(ILLINOIS_TEST, "postfix")
    if KEEP_INFIX_PARENTHESIS:
        test_inf_il = remove_variables(ILLINOIS_TEST)
        test_inf_il = test_inf_il[:len(test_pos_il)]
    else:
        test_inf_il = convert_to(ILLINOIS_TEST, "infix")

    to_binary(os.path.join(DIR_PATH, "../test_il_prefix.p"),
              test_pre_il)
    to_binary(os.path.join(DIR_PATH, "../test_il_postfix.p"),
              test_pos_il)
    to_binary(os.path.join(DIR_PATH, "../test_il_infix.p"),
              test_inf_il)

    # Illinois training data
    pre_il = convert_to(ILLINOIS, "prefix")
    pos_il = convert_to(ILLINOIS, "postfix")
    if KEEP_INFIX_PARENTHESIS:
        inf_il = remove_variables(ILLINOIS)
        inf_il = inf_il[:len(pos_il)]
    else:
        inf_il = convert_to(ILLINOIS, "infix")

    if MAKE_IND_SETS:
        to_binary(os.path.join(DIR_PATH, "../train_il_prefix.p"),
                  pre_il)
        to_binary(os.path.join(DIR_PATH, "../train_il_postfix.p"),
                  pos_il)
        to_binary(os.path.join(DIR_PATH, "../train_il_infix.p"),
                  inf_il)

    # MAWPS testing data
    test_pre_mawps = convert_to(MAWPS_TEST, "prefix")
    test_pos_mawps = convert_to(MAWPS_TEST, "postfix")
    if KEEP_INFIX_PARENTHESIS:
        test_inf_mawps = remove_variables(MAWPS_TEST)
        test_inf_mawps = test_inf_mawps[:len(test_pos_mawps)]
    else:
        test_inf_mawps = convert_to(MAWPS_TEST, "infix")

    to_binary(os.path.join(DIR_PATH, "../test_mawps_prefix.p"),
              test_pre_mawps)
    to_binary(os.path.join(DIR_PATH, "../test_mawps_postfix.p"),
              test_pos_mawps)
    to_binary(os.path.join(DIR_PATH, "../test_mawps_infix.p"),
              test_inf_mawps)

    # MAWPS training data
    pre_mawps = convert_to(MAWPS, "prefix")
    pos_mawps = convert_to(MAWPS, "postfix")
    if KEEP_INFIX_PARENTHESIS:
        inf_mawps = remove_variables(MAWPS)
        inf_mawps = inf_mawps[:len(pos_mawps)]
    else:
        inf_mawps = convert_to(MAWPS, "infix")

    if MAKE_IND_SETS:
        to_binary(os.path.join(DIR_PATH, "../train_mawps_prefix.p"),
                  pre_mawps)
        to_binary(os.path.join(DIR_PATH, "../train_mawps_postfix.p"),
                  pos_mawps)
        to_binary(os.path.join(DIR_PATH, "../train_mawps_infix.p"),
                  inf_mawps)

    if USE_GENERATED:
        # GENERATED testing data
        test_pre_gen = convert_to(GENERATED_TEST, "prefix")
        test_pos_gen = convert_to(GENERATED_TEST, "postfix")
        if KEEP_INFIX_PARENTHESIS:
            test_inf_gen = remove_variables(GENERATED_TEST)
            test_inf_gen = test_inf_gen[:len(test_pos_gen)]
        else:
            test_inf_gen = convert_to(GENERATED_TEST, "infix")

        to_binary(os.path.join(DIR_PATH, "../test_gen_prefix.p"),
                  test_pre_gen)
        to_binary(os.path.join(DIR_PATH, "../test_gen_postfix.p"),
                  test_pos_gen)
        to_binary(os.path.join(DIR_PATH, "../test_gen_infix.p"),
                  test_inf_gen)

        # GENERATED training data
        pre_gen = convert_to(GENERATED, "prefix")
        pos_gen = convert_to(GENERATED, "postfix")
        if KEEP_INFIX_PARENTHESIS:
            inf_gen = remove_variables(GENERATED)
            inf_gen = inf_gen[:len(pos_gen)]
        else:
            inf_gen = convert_to(GENERATED, "infix")

        if MAKE_IND_SETS:
            to_binary(os.path.join(DIR_PATH, "../train_gen_prefix.p"),
                      pre_gen)
            to_binary(os.path.join(DIR_PATH, "../train_gen_postfix.p"),
                      pos_gen)
            to_binary(os.path.join(DIR_PATH, "../train_gen_infix.p"),
                      inf_gen)

    combined_prefix = pre_ai2 + pre_common + pre_il + pre_mawps
    if USE_GENERATED:
        combined_prefix += pre_gen
    random.shuffle(combined_prefix)
    to_binary(os.path.join(DIR_PATH, "../train_all_prefix.p"),
              combined_prefix)

    combined_postfix = pos_ai2 + pos_common + pos_il + pos_mawps
    if USE_GENERATED:
        combined_postfix += pos_gen
    random.shuffle(combined_postfix)
    to_binary(os.path.join(DIR_PATH, "../train_all_postfix.p"),
              combined_postfix)

    combined_infix = inf_ai2 + inf_common + inf_il + inf_mawps
    if USE_GENERATED:
        combined_infix += inf_gen
    random.shuffle(combined_infix)
    combined_infix = remove_variables(combined_infix)
    if not KEEP_INFIX_PARENTHESIS:
        combined_infix = convert_to(combined_infix, "infix")
    to_binary(os.path.join(DIR_PATH, "../train_all_infix.p"),
              combined_infix)

    print("\nCreating a small debugging file...")

    small_data = []

    for p in PROBLEM_LIST[:100]:
        small_data.append(p)

    to_binary(os.path.join(DIR_PATH, "../debug.p"), small_data)

    print("...done.")

    # Remove old data statistic file
    if os.path.isfile(DATA_STATS):
        os.remove(DATA_STATS)

    # Write the information about what data was created
    with open(DATA_STATS, "w") as fh:
        fh.write("Data file information. "
                 + "All of the binaries are described below.\n\n")
        fh.write(f"Testing Split: {TEST_SPLIT * 100}%\n\n")
        fh.write("Original: ")
        fh.write("%d problems\n" % len(PROBLEM_LIST))
        fh.write("Debugging Data: ")
        fh.write("%d problems\n" % len(small_data))
        fh.write("\nGenerated Data: ")
        fh.write("%d problems\n" % len(GENERATED))
        fh.write("\nInfix Data: ")
        fh.write("%d problems\n" % len(CLEAN_INFIX_CONVERTED_PROBLEM_LIST))
        fh.write("Prefix Data: ")
        fh.write("%d problems\n" % len(POLISH_CONVERTED_PROBLEM_LIST))
        fh.write("Postfix Data: ")
        fh.write("%d problems\n" % len(REVERSE_POLISH_CONVERTED_PROBLEM_LIST))
        if MAKE_IND_SETS:
            fh.write("\nAI2 Train: ")
            fh.write("%d problems\n" % len(AI2))
            fh.write("Common Core Train: ")
            fh.write("%d problems\n" % len(COMMONCORE))
            fh.write("Illinois Train: ")
            fh.write("%d problems\n" % len(ILLINOIS))
            fh.write("MAWPS Train: ")
            fh.write("%d problems\n" % len(MAWPS))
            fh.write("Generated MWPs (gen): ")
            fh.write("%d problems\n" % len(GENERATED))
        fh.write("\nAI2 Test (Infix): ")
        fh.write("%d problems\n" % len(test_inf_ai2))
        fh.write("AI2 Test (Prefix): ")
        fh.write("%d problems\n" % len(test_pre_ai2))
        fh.write("AI2 Test (Postfix): ")
        fh.write("%d problems\n" % len(test_pos_ai2))
        fh.write("Common Core Test (Infix): ")
        fh.write("%d problems\n" % len(test_inf_common))
        fh.write("Common Core Test (Prefix): ")
        fh.write("%d problems\n" % len(test_pre_common))
        fh.write("Common Core Test (Postfix): ")
        fh.write("%d problems\n" % len(test_pos_common))
        fh.write("Illinois Test (Infix): ")
        fh.write("%d problems\n" % len(test_inf_il))
        fh.write("Illinois Test (Prefix): ")
        fh.write("%d problems\n" % len(test_pre_il))
        fh.write("Illinois Test (Postfix): ")
        fh.write("%d problems\n" % len(test_pos_il))
        fh.write("MAWPS Test (Infix): ")
        fh.write("%d problems\n" % len(test_inf_mawps))
        fh.write("MAWPS Test (Prefix): ")
        fh.write("%d problems\n" % len(test_pre_mawps))
        fh.write("MAWPS Test (Postfix): ")
        fh.write("%d problems\n" % len(test_pos_mawps))
