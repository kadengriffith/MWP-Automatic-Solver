from .preprocessing import remove_stopwords, lemmatize
from .utils import load_data_from_binary, to_binary
from random import shuffle, randint
from os.path import abspath, exists
from tensorflow import constant, data as tfdata


def masked_pretrain(texts):
    examples = []
    targets = []
    mask_token = "<MASK>"
    num_masked = 3

    for text in texts:
        lst_text = text.split(' ')
        random_masked = []
        excluded_indices = []

        if len(lst_text) < num_masked + (num_masked - 1):
            # If sentence too short
            continue

        for _ in range(num_masked):
            # Pick 3 random tokens, not adjacent to another
            index = randint(0, len(lst_text) - 1)

            tries = 0
            while (not lst_text[index].isalpha() or
                    lst_text[index] == mask_token or
                    index in excluded_indices) and \
                    tries < 10:
                tries += 1
                index = randint(0, len(lst_text) - 1)

            # If valid index found, add adjacent indices to exclusion set
            excluded_indices.append(index)
            excluded_indices.append(max(0, index - 1))
            excluded_indices.append(min(index + 1, len(lst_text) - 1))
            random_masked.append((index, lst_text[index]))
            lst_text[index] = mask_token

        # Sort the tokens by appearance in sentence
        random_masked = sorted(random_masked, key=lambda x: x[0])
        mask_target = [token for _, token in random_masked]

        # Recreate the example with masked tokens
        examples.append(' '.join(lst_text))
        # An order of appearance token sentence
        targets.append(' '.join(mask_target))

    return examples, targets


def imdb(remove_stop_words=True, as_lemmas=True):
    from tensorflow_datasets import load, as_numpy
    from re import sub

    train_english = []

    data = load("imdb_reviews", data_dir=abspath("data/datasets/tensorflow"))

    print("Loaded IMDb sentences.")

    english_data = list(as_numpy(data["train"].concatenate(data["test"])))

    # Produce 577,886 examples of english
    for sentence in english_data:
        # Clean up each document in the data
        text = sentence["text"].decode("utf-8")
        text = sub(r"(<br \/>)+", "\n", text)
        text = sub(r",", " , ", text)
        text = sub(r"\"", " \" ", text)
        text = sub(r"'", " '", text)
        text = sub(r"\. ", " .\n", text)
        text = sub(r"\? ", " ?\n", text)
        text = sub(r"! ", " ! ", text)
        text = sub(r"- ", " - ", text)
        text = sub(r"\+ ", " + ", text)
        text = sub(r"\* ", " * ", text)
        text = sub(r"\/ ", " / ", text)

        for se in text.split('\n'):
            # Unlabelled english text
            train_english.append(se.lower())

    if remove_stop_words:
        train_english = remove_stopwords(train_english)

    if as_lemmas:
        train_english = lemmatize(train_english)

    train_english, train_tar = masked_pretrain(train_english)
    # Convert arrays to TensorFlow constants
    train_eng_const = constant(train_english)
    train_tar_const = constant(train_tar)
    # Turn the constants into TensorFlow Datasets
    return tfdata.Dataset.from_tensor_slices((train_eng_const,
                                              train_tar_const)), len(train_english)


def wikipedia(dump_spec_str, remove_stop_words=False, as_lemmas=True):
    from tensorflow_datasets import as_numpy
    from data.util.classes.WikipediaML import WikipediaML

    train_english = []

    if not exists(abspath("data/datasets/wikipedia/wikipedia.sentences.p")):
        specified_wiki_dump = dump_spec_str.split('_')

        # Download the data if not been downloaded
        data = WikipediaML(language=specified_wiki_dump[1],
                           date=specified_wiki_dump[2],
                           data_dir=abspath(
                               f"data/datasets/wikipedia/{specified_wiki_dump[1]}"),
                           verbose=True).load()

        print("Separating Wikipedia data into sentences...")

        for number, data in enumerate(data):
            content = as_numpy(data["text"])[0].decode("utf-8")
            content = content.split("\n\n")

            for sentence in content:
                train_english.append(sentence)

        if remove_stop_words:
            train_english = remove_stopwords(train_english)

        if as_lemmas:
            train_english = lemmatize(train_english)

        train_english, train_tar = masked_pretrain(train_english)

        to_binary(abspath("data/datasets/wikipedia/wikipedia.sentences.p"),
                  train_english)
        to_binary(abspath("data/datasets/wikipedia/wikipedia.targets.p"),
                  train_tar)
        print("Saved Wikipedia sentences.")
    else:
        train_english = load_data_from_binary(
            abspath("data/datasets/wikipedia/wikipedia.sentences.p")
        )
        train_tar = load_data_from_binary(
            abspath("data/datasets/wikipedia/wikipedia.targets.p")
        )
        print("Loaded Wikipedia sentences.")

    # Convert arrays to TensorFlow constants
    train_eng_const = constant(train_english)
    train_tar_const = constant(train_tar)

    # Turn the constants into TensorFlow Datasets
    return tfdata.Dataset.from_tensor_slices((train_eng_const,
                                              train_tar_const)), len(train_english)


def dolphin18k(remove_stop_words=False, as_lemmas=True):
    # Produces 3,476 examples of english text
    train_english = load_data_from_binary(
        abspath("data/datasets/Dolphin18K/dolphin.pretraining.p")
    )

    print("Loaded Dolphin18k sentences.")

    if remove_stop_words:
        train_english = remove_stopwords(train_english)

    if as_lemmas:
        train_english = lemmatize(train_english)

    train_english, train_tar = masked_pretrain(train_english)

    # Convert arrays to TensorFlow constants
    train_eng_const = constant(train_english)
    train_tar_const = constant(train_tar)

    # Turn the constants into TensorFlow Datasets
    return tfdata.Dataset.from_tensor_slices((train_eng_const,
                                              train_tar_const)), len(train_english)
