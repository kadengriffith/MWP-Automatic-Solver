'''
Author: Kaden Griffith
Description: This script implements a math word problem solver using the 2017 Transformer architecture.
Various pre-training methods have been tested and are configured to run here with hopefully minimal
effort. Please follow the directions in the README.md file. You shouldn't need to edit this file, as
it is controlled by a configuration file.
'''
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds

# The Transformer model
from models.transformer.MultiHeadAttention import MultiHeadAttention
from models.transformer.EncoderLayer import EncoderLayer
from models.transformer.Encoder import Encoder
from models.transformer.DecoderLayer import DecoderLayer
from models.transformer.Decoder import Decoder
from models.transformer.Transformer import Transformer
from models.transformer.CustomSchedule import CustomSchedule
from models.transformer.network import create_masks, loss_function
from data.util.utils import load_data_from_binary, to_binary, get_as_tuple, expressionize, print_epoch
from data.util.classes.NumberTag import NumberTag
from data.util.classes.Scorer import Scorer
from data.util.classes.Logger import Logger
from data.util.classes.WikipediaML import WikipediaML

# Utilities
import re
import os
import sys
import json
import time
import random
import numpy as np

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

if not len(sys.argv) > 1:
    '''
    This program expects a config.json filepath to be passed as the first argument.
    Refer to README.md for instructions on how to construct a config file.
    '''
    raise Exception("Please use a config file.")

with open(os.path.join(DIR_PATH, sys.argv[1]), encoding='utf-8-sig') as fh:
    data = json.load(fh)

settings = dict(data)

DATASET = settings["dataset"]
TEST_SET = settings["test"]
PRETRAIN = settings["pretrain"]
DATA_PATH = os.path.join(DIR_PATH, "data/" + DATASET)
EQUALS_SIGN = False

if len(sys.argv) > 2:
    LOSS_THRESHHOLD = float(sys.argv[2])
else:
    LOSS_THRESHHOLD = 0

# If fine-tuning set this to a str containing the model name
CKPT_MODEL = settings["model"]

# SubwordTextEncoder / TextEncoder
# Will get W2V, GloVe, maybe BERT in if this does poorly
ENCODE_METHOD = tfds.features.text.SubwordTextEncoder
TRAIN_WITH_TAGS = True

# Data constraints
MAX_LENGTH = 60

#|##### TF EXAMPLE ######
#  num_layers = 4
#  d_model = 128
#  dff = 512
#  num_heads = 8
#|#######################
# Hyperparameters
NUM_LAYERS = settings["layers"]
D_MODEL = settings["d_model"]
DFF = settings["dff"]
NUM_HEADS = settings["heads"]
DROPOUT = settings["dropout"]

# Training settings
EPOCHS = settings["epochs"]
BATCH_SIZE = settings["batch"]

# Adam optimizer params
BETA_1 = settings["beta_1"]
BETA_2 = settings["beta_2"]
EPSILON = 1e-9

# Random seed for shuffling the data
SEED = settings["seed"]

if isinstance(CKPT_MODEL, str):
    # If a model name is given train from that model
    CONTINUE_FROM_CKPT = True
else:
    CONTINUE_FROM_CKPT = False

# The name to keep track of any changes
if not CONTINUE_FROM_CKPT:
    MODEL_NAME = f"mwp_{NUM_LAYERS}_{NUM_HEADS}_{D_MODEL}_{DFF}_{int(time.time())}"
else:
    MODEL_NAME = CKPT_MODEL
    CHECKPOINT_PATH = os.path.join(DIR_PATH,
                                   f"models/trained/{CKPT_MODEL}/")

# The checkpoint file where the trained weights will be saved
# Only saves on finish
if not os.path.isdir(f"models/trained/{MODEL_NAME}"):
    os.mkdir(f"models/trained/{MODEL_NAME}")

MODEL_PATH = os.path.join(DIR_PATH,
                          f"models/trained/{MODEL_NAME}/")


TEXT_TOKENIZER_PATH = os.path.join(DIR_PATH,
                                   f"models/tokenizers/{MODEL_NAME}_t.p")
EQUATION_TOKENIZER_PATH = os.path.join(DIR_PATH,
                                       f"models/tokenizers/{MODEL_NAME}_e.p")

ARE_TOKENIZERS_PRESENT = os.path.exists(TEXT_TOKENIZER_PATH) \
    or os.path.exists(EQUATION_TOKENIZER_PATH)

# Set the seed for random
random.seed(SEED)

USER_INPUT = settings["input"]
MIRRORED_STRATEGY = tf.distribute.MirroredStrategy()


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


if __name__ == "__main__":
    print("Starting the MWP Transformer training.")

    train_text = []
    train_equations = []

    if PRETRAIN == "imdb":
        print("Getting pretraining data...\n")
        train_english = []
        train_english_blanks = []

        # Pretrain on unlabelled english text for more in-depth understanding of english
        data = tfds.load("imdb_reviews",
                         data_dir=os.path.join(DIR_PATH,
                                               'data/tensorflow-datasets/'))

        english_data = tfds.as_numpy(data["train"])

        english_data = list(english_data)
        random.shuffle(english_data)

        # Produce ~314041 examples of english
        for sentence in english_data:
            # Clean up each document in the data
            text = sentence["text"].decode("utf-8")
            text = re.sub(r"(<br \/>)+", "\n", text)
            text = re.sub(r",", " , ", text)
            text = re.sub(r"\"", " \" ", text)
            text = re.sub(r"'", " '", text)
            text = re.sub(r"\. ", " .\n", text)
            text = re.sub(r"\? ", " ?\n", text)
            text = re.sub(r"! ", " ! ", text)
            text = re.sub(r"- ", " - ", text)
            text = re.sub(r"\+ ", " + ", text)
            text = re.sub(r"\* ", " * ", text)
            text = re.sub(r"\/ ", " / ", text)

            for se in text.split('\n'):
                # Unlabelled english text
                train_english.append(se.lower())
                train_english_blanks.append("")

        # Convert arrays to TensorFlow constants
        train_eng_const = tf.constant(train_english)
        train_blk_const = tf.constant(train_english_blanks)

        # Turn the constants into TensorFlow Datasets
        english_dataset = tf.data.Dataset.from_tensor_slices((train_eng_const,
                                                              train_blk_const))

        print("...done.\n")
    elif "wikipedia" in PRETRAIN:
        # To use the wikipedia data, set 'pretrain: wikipedia_lang_dumpdate' in your config.
        if not os.path.exists(os.path.join(DIR_PATH, "data/wikipedia.sentences.p")):
            specified_wiki_dump = PRETRAIN.split('_')

            # Download the data if not been downloaded
            data = WikipediaML(language=specified_wiki_dump[1],
                               date=specified_wiki_dump[2],
                               data_dir=f"data/{specified_wiki_dump[1]}_wikipedia").load()

            train_english = []
            train_english_blanks = []

            print("Separating Wikipedia data into sentences...")

            for number, data in enumerate(data):
                content = tfds.as_numpy(data["text"])[0].decode("utf-8")
                content = content.split("\n\n")

                for sentence in content:
                    train_english.append(sentence)
                    train_english_blanks.append("")

            to_binary(os.path.join(DIR_PATH, "data/wikipedia.sentences.p"),
                      train_english)
            # A binary of empty strings... Could be better.
            to_binary(os.path.join(DIR_PATH, "data/wikipedia.blanks.p"),
                      train_english_blanks)
            print("Saved Wikipedia sentences.")
        else:
            train_english = load_data_from_binary(
                os.path.join(DIR_PATH, "data/wikipedia.sentences.p")
            )
            train_english_blanks = load_data_from_binary(
                os.path.join(DIR_PATH, "data/wikipedia.blanks.p")
            )
            print("Loaded Wikipedia sentences.")

        # Convert arrays to TensorFlow constants
        train_eng_const = tf.constant(train_english)
        train_blk_const = tf.constant(train_english_blanks)

        # Turn the constants into TensorFlow Datasets
        english_dataset = tf.data.Dataset.from_tensor_slices((train_eng_const,
                                                              train_blk_const))

        print("...done.")
    elif "dolphin" in PRETRAIN:
        dolphin_dataset = load_data_from_binary(
            os.path.join(DIR_PATH,
                         "data/datasets/Dolphin18K/dolphin.pretraining.p")
        )

        # Convert arrays to TensorFlow constants
        train_eng_const = tf.constant(dolphin_dataset)
        train_blk_const = tf.constant(
            ["" for _ in range(len(dolphin_dataset))]
        )

        # Turn the constants into TensorFlow Datasets
        english_dataset = tf.data.Dataset.from_tensor_slices((train_eng_const,
                                                              train_blk_const))

    print(f"\nTokenizing data from {DATASET}...\n")

    examples = load_data_from_binary(DATA_PATH)

    print(f"Shuffling data with seed: {SEED}\n")
    random.shuffle(examples)

    # Get training examples
    for example in examples:
        try:
            if not TRAIN_WITH_TAGS:
                txt, exp = get_as_tuple(example)

                if not EQUALS_SIGN:
                    train_equations.append(expressionize(exp))
                else:
                    train_equations.append(exp)

                train_text.append(txt)
            else:
                txt, exp = get_as_tuple(example)

                masked_txt, masked_exp, _ = NumberTag(txt, exp).get_masked()

                if not EQUALS_SIGN:
                    train_equations.append(expressionize(masked_exp))
                else:
                    train_equations.append(masked_exp)

                train_text.append(masked_txt)
        except:
            pass

    if PRETRAIN == "imdb" or PRETRAIN == "dolphin" or "wikipedia" in PRETRAIN:
        print(
            f"Set to train with {len(train_english)} examples of unlabelled english.\n"
        )
    else:
        print(f"Set to train with {len(train_text)} examples.\n")

    print("Building vocabulary...\n")

    # Convert arrays to TensorFlow constants
    train_text_const = tf.constant(train_text)
    train_eq_const = tf.constant(train_equations)

    # Turn the constants into TensorFlow Datasets
    training_dataset = tf.data.Dataset.from_tensor_slices((train_text_const,
                                                           train_eq_const))

    if PRETRAIN != False:
        training_dataset = training_dataset.concatenate(english_dataset)

    if not ARE_TOKENIZERS_PRESENT:
        # Create data tokenizers
        tokenizer_txt = ENCODE_METHOD.build_from_corpus((txt.numpy() for txt, _ in training_dataset),
                                                        target_vocab_size=2**13)

        tokenizer_eq = ENCODE_METHOD.build_from_corpus((eq.numpy() for _, eq in training_dataset),
                                                       target_vocab_size=2**13)

        to_binary(os.path.join(DIR_PATH, f"models/tokenizers/{MODEL_NAME}_t.p"),
                  tokenizer_txt)
        to_binary(os.path.join(DIR_PATH, f"models/tokenizers/{MODEL_NAME}_e.p"),
                  tokenizer_eq)
    else:
        # Saving the tokenizers significantly speeds up the script
        tokenizer_txt = load_data_from_binary(
            f"models/tokenizers/{MODEL_NAME}_t.p"
        )
        tokenizer_eq = load_data_from_binary(
            f"models/tokenizers/{MODEL_NAME}_e.p"
        )

        print("\nLoaded tokenizers from file.")

    print("Encoding inputs...")

    def encode(lang1, lang2):
        lang1 = [tokenizer_txt.vocab_size] + tokenizer_txt.encode(
            lang1.numpy()
        ) + [tokenizer_txt.vocab_size + 1]

        lang2 = [tokenizer_eq.vocab_size] + tokenizer_eq.encode(
            lang2.numpy()
        ) + [tokenizer_eq.vocab_size + 1]

        return lang1, lang2

    def tf_encode(txt, eq):
        return tf.py_function(encode, [txt, eq], [tf.int64, tf.int64])

    if PRETRAIN != False:
        # Use only the unlabelled english data
        training_dataset = english_dataset.map(tf_encode)
    else:
        training_dataset = training_dataset.map(tf_encode)

    training_dataset = training_dataset.filter(filter_max_length)

    # Cache the dataset to memory to get a speedup while reading from it.
    training_dataset = training_dataset.cache()

    training_dataset = training_dataset.padded_batch(BATCH_SIZE,
                                                     padded_shapes=([-1], [-1]))

    training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    input_vocab_size = tokenizer_txt.vocab_size + 2
    target_vocab_size = tokenizer_eq.vocab_size + 2

    print("...done.")
    print("\nDefining the Transformer model...")

    # Using the Adam optimizer
    learning_rate = settings["lr"]

    if learning_rate == "scheduled":
        learning_rate = CustomSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=BETA_1,
                                         beta_2=BETA_2,
                                         epsilon=EPSILON)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(
        name="train_acc"
    )

    transformer = Transformer(NUM_LAYERS,
                              D_MODEL,
                              NUM_HEADS,
                              DFF,
                              input_vocab_size,
                              target_vocab_size,
                              DROPOUT)

    print("...done.")
    print("\nTraining...\n")

    # Model saving
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    if CONTINUE_FROM_CKPT:
        # Load last checkpoint
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  CHECKPOINT_PATH,
                                                  max_to_keep=999)
    else:
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  MODEL_PATH,
                                                  max_to_keep=999)

    # If a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint and CONTINUE_FROM_CKPT:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f"Restored from {CHECKPOINT_PATH} checkpoint\n")

    # Set the logger to report to the correct file
    logger = Logger(MODEL_NAME)
    # Log all the settings used in the session
    logger.log(MODEL_NAME)
    logger.log(MODEL_PATH)
    if CONTINUE_FROM_CKPT:
        logger.log(f"Continued from {CHECKPOINT_PATH}")
    logger.log(DATASET)
    logger.log(f"Random Shuffle Seed: {SEED}")
    logger.log(f"\nPretraining: {PRETRAIN}")
    logger.log(f"Epochs: {EPOCHS}")
    logger.log(f"Batch Size: {BATCH_SIZE}")
    logger.log(f"Max Length: {MAX_LENGTH}")
    logger.log(f"Equals Sign: {EQUALS_SIGN}")
    logger.log(f"Layers: {NUM_LAYERS}")
    logger.log(f"Heads: {NUM_HEADS}")
    logger.log(f"Model Depth: {D_MODEL}")
    logger.log(f"Feed Forward Depth: {DFF}")
    logger.log(f"Dropout: {DROPOUT}\n")
    logger.log(f"Learning Rate: {learning_rate}\n")
    logger.log(f"Adam Params: b1 {BETA_1} b2 {BETA_2} e {EPSILON}\n")

    with MIRRORED_STRATEGY.scope():
        @tf.function
        def train_step(inp, tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp,
                                                                             tar_inp)

            with tf.GradientTape() as tape:
                # predictions.shape == (batch_size, seq_len, vocab_size)
                predictions, _ = transformer(inp, tar_inp,
                                             True,
                                             enc_padding_mask,
                                             combined_mask,
                                             dec_padding_mask)

                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss,
                                      transformer.trainable_variables)

            optimizer.apply_gradients(zip(gradients,
                                          transformer.trainable_variables))

            train_loss(loss)
            train_acc(tar_real, predictions)

    # Train
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_acc.reset_states()

        # inp -> MWP, tar -> Equation
        for (batch, (inp, tar)) in enumerate(training_dataset):
            train_step(inp, tar)

            if batch % 10 == 0:
                print_epoch("Epoch {}/{} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                    epoch + 1,
                    EPOCHS,
                    batch,
                    train_loss.result(),
                    train_acc.result()))

        print_epoch("Epoch {}/{} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
            epoch + 1,
            EPOCHS,
            batch,
            train_loss.result(),
            train_acc.result()), clear=True)
        # Save a log of the epoch results
        logger.log(
            f"Epoch {epoch + 1}: loss {train_loss.result()} acc {train_acc.result()}")

        # Calculate the time the epoch took to complete
        # The first epoch seems to take significantly longer than the others
        print(f"Epoch took {int(time.time() - start)}s\n")

        if epoch == (EPOCHS - 1) or (train_loss.result() < LOSS_THRESHHOLD and not PRETRAIN):
            # Save a checkpoint of model weights
            ckpt_save_path = ckpt_manager.save()
            print(f'Saved {MODEL_NAME} to {ckpt_save_path}\n')
            os.remove(os.path.join(DIR_PATH, sys.argv[1]))

            settings["model"] = MODEL_NAME

            with open(os.path.join(DIR_PATH, sys.argv[1]), mode="w") as fh:
                json.dump(settings, fh)

            break

    print("...done.")

    def evaluate(inp_sentence):
        start_token = [tokenizer_txt.vocab_size]
        end_token = [tokenizer_txt.vocab_size + 1]

        # The input is a MWP, hence adding the start and end token
        inp_sentence = start_token + \
            tokenizer_txt.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # The target is an equation, the first word to the transformer should be the
        # equation start token.
        decoder_input = [tokenizer_eq.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input,
                                                                             output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, tokenizer_eq.vocab_size + 1):
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def translate(sentence):
        # Translate from MWP to equation
        result, attention_weights = evaluate(sentence)

        predicted_equation = tokenizer_eq.decode([i for i in result
                                                  if i < tokenizer_eq.vocab_size])

        return predicted_equation

    if isinstance(TEST_SET, str):
        print(f'\nTesting translations...')

        if TEST_SET == "infix":
            sets = ["test_ai_infix.p",
                    "test_cc_infix.p",
                    "test_il_infix.p",
                    "test_mawps_infix.p"]
        elif TEST_SET == "postfix":
            sets = ["test_ai_postfix.p",
                    "test_cc_postfix.p",
                    "test_il_postfix.p",
                    "test_mawps_postfix.p"]
        elif TEST_SET == "prefix":
            sets = ["test_ai_prefix.p",
                    "test_cc_prefix.p",
                    "test_il_prefix.p",
                    "test_mawps_prefix.p"]
        else:
            # e.g. ai_infix
            sets = [f"test_{TEST_SET}.p"]

        for s in sets:
            logger.log(f"\n{s}")

            test_set = load_data_from_binary(os.path.join(DIR_PATH,
                                                          "data/" + s))

            bleu = []
            # Test the model's translations on withheld data
            for i, data in enumerate(test_set):
                data_dict = dict(data)

                q, e = data_dict["question"], data_dict["equation"]

                tagger = NumberTag(q, e)

                clean_q, clean_e = tagger.get_originals()

                masked_input, masked_equation, mask_map = tagger.get_masked()

                predicted = translate(masked_input)

                unmasked_prediction = tagger.apply_map(predicted, mask_map)

                logger.plog(f"Input: {clean_q}")
                logger.plog(f"Hypothesis: {unmasked_prediction}")
                if EQUALS_SIGN:
                    logger.plog(f"Actual:    {clean_e}")

                    bleu.append((unmasked_prediction, clean_e))
                else:
                    logger.plog(f"Actual:    {expressionize(clean_e)}")

                    bleu.append((unmasked_prediction, expressionize(clean_e)))

            n_attempt, perfect_percentage, precision, average_bleu = Scorer(
                bleu).get()

            logger.plog(
                f"\nOut of {n_attempt} predictions, {perfect_percentage}% were correct with {average_bleu} Bleu-2 and {precision}% average precision.\n"
            )

        print("...done.")

    while True and EPOCHS == 0 and USER_INPUT:
        # Testing live really doesn't work all that great.
        # It's fun to see the system in action though.
        inp = input("Enter a MWP > ")

        tagger = NumberTag(inp, "")
        masked_input, _, mask_map = tagger.get_masked()

        predicted = translate(masked_input)
        unmasked_prediction = tagger.apply_map(predicted, mask_map)

        print(f"Possible Equation: {unmasked_prediction}")

    print("Exiting.")
