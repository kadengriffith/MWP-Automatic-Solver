'''
Author: Kaden Griffith
Description: This script implements a math word problem solver using the 2017 Transformer architecture.
Various pre-training methods have been tested and are configured to run here with hopefully minimal
effort. Please follow the directions in the README.md file. You shouldn't need to edit this file, as
it is controlled by a configuration file.
'''
from __future__ import absolute_import, unicode_literals, division, print_function
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

# Utilities
import data.util.utils as utils
import data.util.preprocessing as preprocessing
import data.util.pretraining as pretraining
from data.util.classes.NumberTag import NumberTag
from data.util.classes.Scorer import Scorer
from data.util.classes.Logger import Logger
from time import time
from random import seed, shuffle
from os import makedirs, remove, environ
from os.path import abspath, exists, join
from re import match
import logging
import yaml
import sys

# Disable TF logging. Turn this on if you are getting unexpected behavior!
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if not len(sys.argv) > 1:
    '''
    This program expects a config.json filepath to be passed as the first argument.
    Refer to README.md for instructions on how to construct a config file.
    '''
    raise Exception("Please use a config file.")

with open(abspath(sys.argv[1]), 'r', encoding='utf-8-sig') as yaml_file:
    settings = yaml.load(yaml_file, Loader=yaml.FullLoader)

DATASET = settings["dataset"]
DUPLICATION = settings["duplication"]
TEST_SET = settings["test"]
PRETRAIN = settings["pretrain"]
TRAIN_WITH_TAGS = True
REORDER_SENTENCES = settings["reorder"]
REMOVE_NUMBERS_NOT_IN_EQ = settings["tagging"]
REMOVE_STOP_WORDS = settings["remove_stopwords"]
PART_OF_SPEECH_TAGGING = settings["pos"]
PART_OF_SPEECH_TAGGING_W_WORDS = settings["pos_words"]
LEMMAS = settings["as_lemmas"]
DATA_PATH = abspath("data/" + DATASET)
USER_INPUT = settings["input"]
EQUALS_SIGN = False
SAVE = settings["save"]
utils.MAX_LENGTH = 60

if len(sys.argv) > 2:
    LOSS_THRESHHOLD = float(sys.argv[2])
else:
    LOSS_THRESHHOLD = 0

# If fine-tuning set this to a str containing the model name
CKPT_MODEL = settings["model"]

# text.SubwordTextEncoder / text.TextEncoder
ENCODE_METHOD = tfds.features.text.SubwordTextEncoder
MIRRORED_STRATEGY = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.ReductionToOneDevice()
)

# Hyperparameters
NUM_LAYERS = settings["layers"]
D_MODEL = settings["d_model"]
DFF = settings["dff"]
NUM_HEADS = settings["heads"]
DROPOUT = settings["dropout"]

# Training settings
EPOCHS = settings["epochs"]
BATCH_SIZE = settings["batch"]
GLOBAL_BATCH_SIZE = BATCH_SIZE * MIRRORED_STRATEGY.num_replicas_in_sync


# Adam optimizer params
BETA_1 = settings["beta_1"]
BETA_2 = settings["beta_2"]
EPSILON = 1e-9

LIVE_MODE = EPOCHS == 0 and USER_INPUT

# Random seed for shuffling the data
SEED = settings["seed"]
# Set the seed for random
seed(SEED)
tf.compat.v1.set_random_seed(SEED)

if isinstance(CKPT_MODEL, str):
    # If a model name is given train from that model
    CONTINUE_FROM_CKPT = True
    MODEL_NAME = CKPT_MODEL
    CHECKPOINT_PATH = abspath(f"models/trained/{CKPT_MODEL}/")
else:
    CONTINUE_FROM_CKPT = False
    MODEL_NAME = f"mwp_{NUM_LAYERS}_{NUM_HEADS}_{D_MODEL}_{DFF}_{int(time())}"

TRAINED_PATH = abspath(f"models/trained/")
MODEL_PATH = abspath(f"models/trained/{MODEL_NAME}/")

TEXT_TOKENIZER_PATH = abspath(
    f"models/trained/{MODEL_NAME}/tokenizers/{MODEL_NAME}_t.p")
EQUATION_TOKENIZER_PATH = abspath(
    f"models/trained/{MODEL_NAME}/tokenizers/{MODEL_NAME}_e.p")

ARE_TOKENIZERS_PRESENT = exists(TEXT_TOKENIZER_PATH) \
    or exists(EQUATION_TOKENIZER_PATH)

tf.compat.v1.enable_eager_execution()

if __name__ == "__main__":
    if LIVE_MODE:
        print("Starting the MWP Transformer live testing.")
    else:
        print("Starting the MWP Transformer training.")

    if not exists(TRAINED_PATH):
        makedirs(TRAINED_PATH)

    num_examples = 0
    if not isinstance(PRETRAIN, bool) and not LIVE_MODE:
        print("Getting pre-training data...")
        if PRETRAIN == "imdb":
            # Pretrain on unlabelled english text for more in-depth understanding of english
            english_dataset, num_examples = pretraining.imdb(remove_stop_words=REMOVE_STOP_WORDS,
                                                             as_lemmas=LEMMAS)
        elif PRETRAIN == "dolphin":
            english_dataset, num_examples = pretraining.dolphin18k(remove_stop_words=REMOVE_STOP_WORDS,
                                                                   as_lemmas=LEMMAS)
        elif "wikipedia" in PRETRAIN:
            # To use the wikipedia data, set 'pretrain: wikipedia_lang_dumpdate' in your config.
            english_dataset, num_examples = pretraining.wikipedia(PRETRAIN,
                                                                  remove_stop_words=REMOVE_STOP_WORDS,
                                                                  as_lemmas=LEMMAS)
            # Take a portion that can be handled by our memory and time restrictions
            limiter = 1000000
            num_examples = limiter
            english_dataset = english_dataset.shuffle(10000).take(limiter)
        else:
            raise Exception(
                "Invalid pre-train setting. Please refer to README.md for help."
            )
        print("...done.")

    print(f"Tokenizing data from {DATASET}...")

    examples = utils.load_data_from_binary(DATA_PATH)

    print(f"Shuffling data with seed: {SEED}")
    shuffle(examples)

    # Get training examples
    train_text = []
    train_equations = []
    incorrect = []
    c = 0
    for example in examples:
        txt, exp = utils.get_as_tuple(example)
        old1, old2 = txt, exp

        if not EQUALS_SIGN:
            exp = utils.expressionize(exp)

        if REMOVE_NUMBERS_NOT_IN_EQ == "st":
            txt = preprocessing.selective_tagging(txt, exp)
        elif REMOVE_NUMBERS_NOT_IN_EQ == "lst":
            txt = preprocessing.label_selective_tagging(txt)
        elif REMOVE_NUMBERS_NOT_IN_EQ == "et":
            txt = preprocessing.exclusive_tagging(txt, exp)

        if not TRAIN_WITH_TAGS:
            train_text.append(txt)
            train_equations.append(exp)
        else:
            tagger = NumberTag(txt, exp)
            masked_txt, masked_exp, _ = tagger.get_masked()

            if tagger.mapped_correctly():
                train_text.append(masked_txt)
                train_equations.append(masked_exp)
            else:
                incorrect.append((old1,
                                  txt,
                                  masked_txt,
                                  old2,
                                  exp,
                                  masked_exp))

    if REMOVE_STOP_WORDS and not PART_OF_SPEECH_TAGGING:
        # Remove all stop words from texts
        print("Removing stop words...")
        train_text = preprocessing.remove_stopwords(train_text)
        print("...done.")

    if PART_OF_SPEECH_TAGGING:
        if REMOVE_STOP_WORDS:
            # Remove all stop words from texts and POS tag
            print("Removing stop words and applying POS tagging...")
            train_text = preprocessing.pos_tag_all(train_text,
                                                   with_word=PART_OF_SPEECH_TAGGING_W_WORDS,
                                                   include_stop_words=False)
            print("...done.")
        else:
            print("Applying POS tagging...")
            train_text = preprocessing.pos_tag_all(train_text,
                                                   with_word=PART_OF_SPEECH_TAGGING_W_WORDS)
            print("...done.")

    # Set the logger to report to the correct file
    logger = Logger(MODEL_NAME)
    logger.plog(
        f"{len(incorrect)}/{len(examples)} of MWP questions were not mapped correctly."
    )
    logger.plog(
        f"{len(train_text)}/{len(examples)} of MWP questions are being used."
    )
    logger.plog(f"{num_examples} English examples are being used.")

    if not LIVE_MODE:
        if REORDER_SENTENCES != False:
            print("Reordering question sentences...")
            train_text, train_equations = preprocessing.reorder_sentences(train_text,
                                                                          train_equations)
            print(
                f"After reordering, {len(train_text)} examples are being used.")
            print("...done.")

        if DUPLICATION != False:
            print(f"Applying duplication factor: {DUPLICATION}...")
            # Duplicate
            train_text_base = train_text.copy()
            train_eq_base = train_equations.copy()

            for i in range(max(0, DUPLICATION - 1)):
                train_text += (reversed(train_text_base),
                               train_text_base)[i % 2 == 0]
                train_equations += train_eq_base

            print(f"Data upscaled to {len(train_text)} examples.")
            print("...done.")

        if PRETRAIN != False and \
                (PRETRAIN in ["imdb", "dolphin"] or "wikipedia" in PRETRAIN):
            print(f"Set to train with {num_examples} examples of English.")
        else:
            print(f"Set to train with {len(train_text)} MWP examples.")

    print("Building vocabulary and encoding data...")

    # Convert arrays to TensorFlow constants
    train_text_const = tf.constant(train_text)
    train_eq_const = tf.constant(train_equations)

    # Turn the constants into TensorFlow Datasets
    training_dataset = tf.data.Dataset.from_tensor_slices((train_text_const,
                                                           train_eq_const))

    if PRETRAIN != False and not LIVE_MODE:
        training_dataset = training_dataset.concatenate(english_dataset)

    if not ARE_TOKENIZERS_PRESENT:
        # Create data tokenizers
        tokenizer_txt = ENCODE_METHOD.build_from_corpus((txt.numpy() for txt, _ in training_dataset),
                                                        target_vocab_size=2**13)

        tokenizer_eq = ENCODE_METHOD.build_from_corpus((eq.numpy() for _, eq in training_dataset),
                                                       target_vocab_size=2**13)

        makedirs(join(MODEL_PATH, "tokenizers"))

        utils.to_binary(TEXT_TOKENIZER_PATH, tokenizer_txt)
        utils.to_binary(EQUATION_TOKENIZER_PATH, tokenizer_eq)
    else:
        # Saving the tokenizers significantly speeds up the script
        tokenizer_txt = utils.load_data_from_binary(TEXT_TOKENIZER_PATH)
        tokenizer_eq = utils.load_data_from_binary(EQUATION_TOKENIZER_PATH)
        print("Loaded tokenizers from file.")

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

    training_dataset = training_dataset.filter(utils.filter_max_length)

    # Cache the dataset to memory to get a speedup while reading from it.
    training_dataset = training_dataset.cache()

    training_dataset = training_dataset.padded_batch(GLOBAL_BATCH_SIZE,
                                                     padded_shapes=([-1], [-1]))

    training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Distribute to GPUs
    training_dataset = MIRRORED_STRATEGY.experimental_distribute_dataset(
        training_dataset
    )

    data_iter = iter(training_dataset)

    input_vocab_size = tokenizer_txt.vocab_size + 2
    target_vocab_size = tokenizer_eq.vocab_size + 2

    print("...done.")
    print("Defining the Transformer model...")

    # Using the Adam optimizer
    learning_rate = settings["lr"]

    if learning_rate == "scheduled":
        learning_rate = CustomSchedule(D_MODEL)

    with MIRRORED_STRATEGY.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate,
                                             beta_1=BETA_1,
                                             beta_2=BETA_2,
                                             epsilon=EPSILON)

        train_acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy'
        )

        train_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
        )

        def compute_loss(labels, predictions):
            per_example_loss = train_loss(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss,
                                              global_batch_size=GLOBAL_BATCH_SIZE)

        transformer = Transformer(NUM_LAYERS,
                                  D_MODEL,
                                  NUM_HEADS,
                                  DFF,
                                  input_vocab_size,
                                  target_vocab_size,
                                  DROPOUT)

        # Model saving
        ckpt = tf.train.Checkpoint(transformer=transformer,
                                   optimizer=optimizer)

    print("...done.")
    print("Training...")

    if CONTINUE_FROM_CKPT:
        # Load last checkpoint
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  CHECKPOINT_PATH,
                                                  max_to_keep=1)

        # If a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print(f"Restored from {CHECKPOINT_PATH} checkpoint!")
    else:
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  MODEL_PATH,
                                                  max_to_keep=1)

    with MIRRORED_STRATEGY.scope():
        def train_step(data_batch):
            inp, tar = data_batch
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

                loss = compute_loss(tar_real, predictions)

            gradients = tape.gradient(loss,
                                      transformer.trainable_variables)

            optimizer.apply_gradients(zip(gradients,
                                          transformer.trainable_variables))

            train_acc.update_state(tar_real, predictions)

            return loss

        # https://github.com/tensorflow/tensorflow/issues/32232
        @tf.function(input_signature=[data_iter.element_spec], experimental_relax_shapes=True)
        def distributed_train_step(diter_next):
            per_replica_losses = MIRRORED_STRATEGY.experimental_run_v2(train_step,
                                                                       args=(diter_next,))
            return MIRRORED_STRATEGY.reduce(tf.distribute.ReduceOp.SUM,
                                            per_replica_losses,
                                            axis=None)

    # The checkpoint file where the trained weights will be saved
    # Only saves on finish
    if not exists(MODEL_PATH):
        makedirs(MODEL_PATH)

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
    logger.log(f"Max Length: {utils.MAX_LENGTH}")
    logger.log(f"Equals Sign: {EQUALS_SIGN}")
    logger.log(f"Layers: {NUM_LAYERS}")
    logger.log(f"Heads: {NUM_HEADS}")
    logger.log(f"Model Depth: {D_MODEL}")
    logger.log(f"Feed Forward Depth: {DFF}")
    logger.log(f"Dropout: {DROPOUT}\n")
    logger.log(f"Learning Rate: {learning_rate}\n")
    logger.log(f"Adam Params: b1 {BETA_1} b2 {BETA_2} e {EPSILON}\n")
    logger.log(f"POS: {PART_OF_SPEECH_TAGGING}")
    logger.log(f"WPOS: {PART_OF_SPEECH_TAGGING_W_WORDS}")
    logger.log(f"SW: {REMOVE_STOP_WORDS}")
    logger.log(f"L: {LEMMAS}")
    logger.log(f"R: {REORDER_SENTENCES}")
    logger.log(f"Tagging: {REMOVE_NUMBERS_NOT_IN_EQ}\n")

    # Don't save on high loss, but don't miss making a checkpoint
    best_loss = 0.5
    model_recorded = False

    # Train
    with MIRRORED_STRATEGY.scope():
        for epoch in range(EPOCHS):
            printable_epoch = epoch + 1
            start = time()

            total_loss = 0.0
            # inp: MWP, tar: Equation
            data_iter = iter(training_dataset)
            for batch, _ in enumerate(training_dataset):
                total_loss += distributed_train_step(next(data_iter))

                if batch % 100 == 0:
                    utils.print_epoch(
                        f"Epoch {printable_epoch}/{EPOCHS} Batch {batch}; Loss {total_loss / batch:.4}; Accuracy {train_acc.result():.4f};{' ' * 20}"
                    )

            current_loss = total_loss / batch
            current_accuracy = train_acc.result()

            utils.print_epoch(f"Epoch {printable_epoch}/{EPOCHS} Batch {batch}; Loss {current_loss:.4}; Accuracy {current_accuracy:.4f};{' ' * 20}",
                              clear=True)

            # Save a log of the epoch results
            logger.log(
                f"Epoch {printable_epoch}: Loss {current_loss}; Accuracy {current_accuracy};"
            )

            # Calculate the time the epoch took to complete
            # The first epoch seems to take significantly longer than the others
            time_taken = int(time() - start)
            if time_taken > 5:
                print(f"Epoch took {time_taken}s")

            if best_loss > current_loss or (printable_epoch == EPOCHS and best_loss > current_loss) \
                or PRETRAIN != False \
                    and SAVE:
                # Update the best loss
                best_loss = current_loss

                # Save a checkpoint of model weights
                ckpt_save_path = ckpt_manager.save()
                logger.plog(f"Saved {MODEL_NAME} to {ckpt_save_path}!")

                if not model_recorded:
                    # Rewrite the config with new model only if not already updated
                    model_recorded = True

                    with open(abspath(sys.argv[1]), mode="r") as fh:
                        config_file = fh.readlines()

                    remove(abspath(sys.argv[1]))

                    with open(abspath(sys.argv[1]), mode="w") as fh:
                        for line in config_file:
                            if match(r"model:", line) is not None:
                                fh.write(f"model: {MODEL_NAME}\n")
                            else:
                                fh.write(line)

            if current_loss < LOSS_THRESHHOLD and not PRETRAIN:
                # Stop if minimum loss has been met
                break

            train_acc.reset_states()

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

        for i in range(utils.MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input,
                                                                             output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # Select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # Return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, tokenizer_eq.vocab_size + 1):
                return tf.squeeze(output, axis=0), attention_weights

            # Concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def translate(sentence):
        # Translate from MWP to equation
        result, attention_weights = evaluate(sentence)

        predicted_equation = tokenizer_eq.decode([i for i in result
                                                  if i < tokenizer_eq.vocab_size])

        return predicted_equation

    if isinstance(TEST_SET, str) and not LIVE_MODE:
        print(f'Testing translations...')

        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f"Restored from {MODEL_PATH} checkpoint!")

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

        summary = []
        for s in sets:
            logger.log(f"\n{s}")

            test_set = utils.load_data_from_binary(abspath("data/" + s))
            test_text = []
            test_eq = []
            incorrect = []
            for example in test_set:
                txt, exp = utils.get_as_tuple(example)

                if not EQUALS_SIGN:
                    exp = utils.expressionize(exp)

                if REMOVE_NUMBERS_NOT_IN_EQ == "lst":
                    txt = preprocessing.label_selective_tagging(txt)

                test_text.append(txt)
                test_eq.append(exp)

            if REORDER_SENTENCES or isinstance(REORDER_SENTENCES, str):
                test_text, test_eq = preprocessing.reorder_sentences(test_text,
                                                                     test_eq)

            if REMOVE_STOP_WORDS and not PART_OF_SPEECH_TAGGING:
                # Remove all stop words from texts
                test_text = preprocessing.remove_stopwords(test_text)

            if PART_OF_SPEECH_TAGGING:
                if REMOVE_STOP_WORDS:
                    # Remove all stop words from texts and POS tag
                    test_text = preprocessing.pos_tag_all(test_text,
                                                          with_word=PART_OF_SPEECH_TAGGING_W_WORDS,
                                                          include_stop_words=False)
                else:
                    test_text = preprocessing.pos_tag_all(test_text,
                                                          with_word=PART_OF_SPEECH_TAGGING_W_WORDS)

            if LEMMAS:
                # Make words into lemmas
                test_text = preprocessing.lemmatize(test_text)

            bleu = []
            # Test the model's translations on withheld data
            for i, q in enumerate(test_text):
                e = test_eq[i]

                if not EQUALS_SIGN:
                    e = utils.expressionize(e)

                if TRAIN_WITH_TAGS:
                    tagger = NumberTag(q, e)
                    masked_txt, masked_exp, mask_map = tagger.get_masked()
                    _, e = tagger.get_originals()

                    if tagger.mapped_correctly():
                        predicted = translate(masked_txt)
                        predicted = tagger.apply_map(predicted, mask_map)
                    else:
                        incorrect.append((old1,
                                          txt,
                                          masked_txt,
                                          old2,
                                          exp,
                                          masked_exp))
                        continue
                else:
                    predicted = translate(q)

                logger.plog(f"Input: {q}")
                logger.plog(f"Hypothesis: {predicted} {mask_map}")
                logger.plog(f"Actual:     {e}")
                bleu.append((predicted, e))

            n_attempt, perfect_percentage, precision, average_bleu = Scorer(
                bleu).get()

            summary.append(
                f"{s}\nOut of {n_attempt} predictions, {perfect_percentage}% were correct with {average_bleu} Bleu-2 and {precision}% average precision.\n")
            logger.plog(f"{perfect_percentage}% correct")
            logger.plog('-' * 25)

        for s in summary:
            logger.plog(s)

        if len(incorrect):
            print(incorrect)
        print("...done.")

    if LIVE_MODE:
        while True:
            # Testing live really doesn't work all that great.
            # It's fun to see the system in action though.
            inp = input("Enter a MWP > ")

            tagger = NumberTag(inp, "")
            masked_input, _, mask_map = tagger.get_masked()

            predicted = translate(masked_input)
            unmasked_prediction = tagger.apply_map(predicted, mask_map)

            print(f"Possible Equation: {unmasked_prediction}")

    print("Exiting.")
