from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import download, data, pos_tag
from collections import Counter
from itertools import permutations
from os.path import abspath, exists
from re import match, sub

NLTK_DIR = abspath("data/datasets/nltk/")
data.path.append(NLTK_DIR)

# Download NLTK data if not already existing
if not exists(abspath("data/datasets/nltk/corpora/stopwords/english")):
    download("stopwords",
             download_dir=NLTK_DIR)

if not exists(abspath("data/datasets/nltk/corpora/wordnet")):
    download("wordnet",
             download_dir=NLTK_DIR)

if not exists(abspath("data/datasets/nltk/tokenizers/punkt")):
    download("punkt",
             download_dir=NLTK_DIR)

if not exists(abspath("data/datasets/nltk/taggers/averaged_perceptron_tagger")):
    download("averaged_perceptron_tagger",
             download_dir=NLTK_DIR)

STOPWORDS = set(stopwords.words('english'))
NOUNS = {x.name().split('.', 1)[0]
         for x in wordnet.all_synsets('n')}
WNL = WordNetLemmatizer()


def is_number(word):
    try:
        float(word)
    except:
        return False
    return True


def pos_tag_all(lst, with_word=True, include_stop_words=True):
    '''
    Use part of speech tagging to alter perceptions of vocabulary patterns in questions.
    '''
    output_array = []
    for sentence in lst:
        temp_list = []
        for i in sent_tokenize(sentence):
            word_list = i.split()
            if not include_stop_words:
                word_list = [w for w in word_list if not w in STOPWORDS]

            tag_indices = {}
            for j, word in enumerate(word_list):
                if match(r"\<.*\>", word) is not None:
                    # Word is a tagged number
                    tag_indices[j] = word

            for j, (word, pos) in enumerate(pos_tag(word_list)):
                if j in tag_indices.keys():
                    temp_list.append(tag_indices[j])
                elif is_number(word):
                    temp_list.append(word)
                else:
                    if with_word:
                        temp_list.append(f"({word} {pos})")
                    else:
                        temp_list.append(f"{pos}")

            pos_sentence = ' '.join(temp_list)
            pos_sentence = sub(r"\.\s+\.", '.', pos_sentence)
            pos_sentence = sub(r"\?\s+\.", '?', pos_sentence)
            pos_sentence = sub(r"\$\s+\$", '$', pos_sentence)

        output_array.append(pos_sentence)
    return output_array


def lemmatize(lst):
    '''
    Convert plurals to lemmas in questions for consistent language.
    '''
    output_array = []
    for sentence in lst:
        temp_list = []
        for word in sentence.split(' '):
            temp_list.append(WNL.lemmatize(word))
        output_array.append(' '.join(temp_list))
    return output_array


def remove_stopwords(lst):
    '''
    Remove any stop words in the question.
    '''
    output_array = []
    for sentence in lst:
        temp_list = []
        for word in sentence.split(' '):
            if word.lower() not in STOPWORDS:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array


def selective_tagging(text, equation):
    '''
    Force any number in the question not appearing in the equation to not be tagged.
    '''

    relevant_numbers = []
    output = []
    for term in equation.split(' '):
        try:
            relevant_numbers.append(float(term))
        except:
            pass

    for word in text.split(' '):
        try:
            # word is a number
            if float(word) in relevant_numbers:
                output.append(word)
            else:
                output.append(f"[[{word}]]")
        except:
            output.append(word)

    return ' '.join(output)


def unique(lst):
    unique_list = []
    for x in lst:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def count_numbers(text):
    c = 0
    for x in text.split(' '):
        try:
            x = float(x)
            c += 1
        except:
            pass
    return c


def label_selective_tagging(text):
    '''
    Remove any numbers that are not labelled as a subject
    '''

    if count_numbers(text) > 2:
        sentence = text.split(' ')
        labels = Counter(sentence).items()
        labels = sorted(labels, key=lambda item: item[1], reverse=True)
        label_candidates = []
        for lbl, freq in labels:
            if freq > 1 and not lbl in STOPWORDS:
                label_candidates.append(lbl)

        window = 4
        output = []
        for j, word in enumerate(sentence):
            try:
                # word is a number
                n = float(word)
                lookahead = sentence[j + 1:min(j + window + 1, len(sentence))]
                lookbehind = sentence[max(j - window, 0):min(j, len(sentence))]

                if lookahead[0] in ["are", "have", "were", "is", "with", "of"]:
                    output.append(word)
                    continue

                if lookahead[0] != label_candidates[0]:
                    continue

                if lookbehind[-1] in ["has", "holds", "contains", "$"]:
                    output.append(word)
                    continue

                if lookbehind[-1] in ["and"] and not lookahead[0] == label_candidates[0]:
                    continue

                move_on = False
                for lbl in lookahead:
                    if '.' == lbl:
                        break

                    if lbl == label_candidates[0]:
                        output.append(word)
                        move_on = True
                        break

                if move_on:
                    continue

                for lbl in lookbehind:
                    if lbl in label_candidates[:1]:
                        output.append(word)
                        break
            except:
                output.append(word)

        o = ' '.join(output)

        if count_numbers(o) >= 2:
            return o
        else:
            return text
    else:
        return text


def reorder_sentences(texts, equations):
    examples = []
    for i, text in enumerate(texts):
        examples.append((text, equations[i]))

    new_texts = []
    new_equations = []

    for e in examples:
        split = e[0].split(" . ")
        if len(split) > 2:
            text_sections = split[:-1]
            text_section_possibilities = list(permutations(text_sections))

            for ps in text_section_possibilities:
                s = list(ps)
                s.append(split[-1])
                new_texts.append(" . ".join(s))
                new_equations.append(e[1])

    return new_texts, new_equations


def exclusive_tagging(text, equation):
    '''
    Remove any number in the question not appearing in the equation.
    '''
    relevant_numbers = []
    output = []
    for term in equation.split(' '):
        try:
            relevant_numbers.append(float(term))
        except:
            pass

    for word in text.split(' '):
        try:
            # word is a number
            if float(word) in relevant_numbers:
                output.append(word)
        except:
            output.append(word)

    return ' '.join(output)
