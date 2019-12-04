from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
# The Transformer model
from models.transformer.MultiHeadAttention import MultiHeadAttention
from models.transformer.EncoderLayer import EncoderLayer
from models.transformer.Encoder import Encoder
from models.transformer.DecoderLayer import DecoderLayer
from models.transformer.Decoder import Decoder
from models.transformer.Transformer import Transformer
from models.transformer.CustomSchedule import CustomSchedule
from models.transformer.network import create_masks
from utils import load_data_from_binary
from classes.NumberTag import NumberTag
from classes.Logger import Logger
# Utilities
import time
import os
import sys
import json

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

if not len(sys.argv) > 1:
    raise Exception("Please use a config file.")

with open(os.path.join(DIR_PATH, sys.argv[1]), encoding='utf-8-sig') as fh:
    data = json.load(fh)

settings = dict(data)

# The model we're testing
CKPT_MODEL = settings["model"]

# Data constraints
MAX_LENGTH = 60

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

if not isinstance(CKPT_MODEL, str):
    raise Exception("No checkpoint could be loaded.")

MODEL_NAME = CKPT_MODEL
CHECKPOINT_PATH = os.path.join(DIR_PATH,
                               f"../models/trained/{CKPT_MODEL}/")

if __name__ == "__main__":
    print("Starting the MWP Transformer time test.")

    tokenizer_txt = load_data_from_binary(
        f"../models/tokenizers/{MODEL_NAME}_t.pickle")
    tokenizer_eq = load_data_from_binary(
        f"../models/tokenizers/{MODEL_NAME}_e.pickle")

    print("Loaded tokenizers from file.")

    input_vocab_size = tokenizer_txt.vocab_size + 2
    target_vocab_size = tokenizer_eq.vocab_size + 2

    print("Defining the Transformer model...")

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
        name="train_acc")

    transformer = Transformer(NUM_LAYERS,
                              D_MODEL,
                              NUM_HEADS,
                              DFF,
                              input_vocab_size,
                              target_vocab_size,
                              DROPOUT)

    print("...done.")
    print("\nLoading checkpoint...\n")

    # Model saving
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    # Load last checkpoint
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              CHECKPOINT_PATH,
                                              max_to_keep=999)

    # If a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f"Restored {CHECKPOINT_PATH} checkpoint!\n")

    # Set the logger to report to the correct file
    logger = Logger(MODEL_NAME)

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

    # The four questions reported in the paper
    test_set = ["There are 22 walnut trees currently in the park . Park workers will plant walnut trees today . When the workers are finished there will be 55 walnut trees in the park . How many walnut trees did the workers plant today ?",
                "Roger had 16 dollars . For his birthday he got 28 more dollars but spent 25 on a new game . How much money does he have now ?",
                "Brenda starts with 7 Skittles . She buys 8 more . How many Skittles does Brenda end with ?",
                "One pencil weighs 28.3 grams . How much do 5 pencils weigh ?"]

    # A list we can average
    times = []

    for question in test_set:
        start = time.time()

        tagger = NumberTag(question, "")

        clean_q, _ = tagger.get_originals()

        masked_input, _, mask_map = tagger.get_masked()

        predicted = translate(masked_input)

        # The network is done once it predicts the expression
        hypothesis = tagger.apply_map(predicted, mask_map)

        # Record the time taken in seconds
        times.append(time.time() - start)

    average_time = sum(times) / len(times)

    for i, question in enumerate(test_set):
        logger.plog(
            f"Question Answered:\n{question}\nTime Taken:\n{times[i]}s | {times[i] * 1000}ms | {times[i] * 1000000}µs")

    logger.plog(
        f"The average time for {MODEL_NAME}:\n{average_time}s | {average_time * 1000}ms | {average_time * 1000000}µs")

    print("...done.")
    print("Exiting.")
