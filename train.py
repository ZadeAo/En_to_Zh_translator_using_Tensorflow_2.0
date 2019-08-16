from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import io
import re
import time
import jieba
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from config import *


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(
            hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


def preprocess_english(w):
    w = re.sub(r"([?.!,])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.lower().strip()
    w = '<start> ' + w + ' <end>'
    return w


def train():
    path_to_zip = keras.utils.get_file(
        'en_zh.zip', origin='https://yun.yusanshi.com/TF_datasets/en_zh.zip', extract=True)
    paths = [os.path.join(os.path.dirname(path_to_zip), 'manythings_org_anki.tsv'),
             os.path.join(os.path.dirname(path_to_zip), 'news-commentary-v14.en-zh.tsv')]

    def preprocess_chinese(w):
        w = ' '.join(jieba.cut(w))
        w = '<start> ' + w + ' <end>'
        return w

    def create_dataset(paths, num_examples):
        lines = []
        for path in paths:
            lines.extend(
                io.open(path, encoding='utf-8').read().strip().split('\n'))
        pairs = [l.split('\t') for l in lines[:num_examples]]
        processed_pairs = [
            [preprocess_english(pair[0]), preprocess_chinese(pair[1])] for pair in pairs]
        return zip(*processed_pairs)

    def max_length(tensor):
        return max(len(t) for t in tensor)

    def tokenize(lang):
        lang_tokenizer = keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = keras.preprocessing.sequence.pad_sequences(
            tensor, padding='post')
        return tensor, lang_tokenizer

    def load_dataset(paths, num_examples=None):
        input_lang, target_lang = create_dataset(paths, num_examples)
        input_tensor, input_lang_tokenizer = tokenize(input_lang)
        target_tensor, target_lang_tokenizer = tokenize(target_lang)
        return input_tensor, input_lang_tokenizer, target_tensor, target_lang_tokenizer

    input_tensor, input_lang_tokenizer, target_tensor, target_lang_tokenizer = load_dataset(
        paths, NUM_EXAMPLES)

    max_input_length = max_length(input_tensor)
    max_target_length = max_length(target_tensor)

    # input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
    #     input_tensor, target_tensor, test_size=0.2)
    input_tensor_train = input_tensor
    target_tensor_train = target_tensor

    BUFFER_SIZE = len(input_tensor_train)
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
    vocab_inp_size = len(input_lang_tokenizer.word_index) + 1
    vocab_tar_size = len(target_lang_tokenizer.word_index) + 1

    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims(
                [target_lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(
                    dec_input, dec_hidden, enc_output)

                loss += loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))

        print('Epoch {} Loss {:.4f}'.format(
            epoch + 1, total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def save_models(variables, models):
        for k, v in variables.items():
            with open(os.path.join(MODEL_PATH, k+'.pickle'), 'wb') as handle:
                pickle.dump(v, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for k, v in models.items():
            v.save_weights(os.path.join(MODEL_PATH, k+'.ckpt'))

        print('Model saved in %s.' % MODEL_PATH)

    save_models(
        {
            'input_lang_tokenizer': input_lang_tokenizer,
            'target_lang_tokenizer': target_lang_tokenizer,
            'max_input_length': max_input_length,
            'max_target_length': max_target_length,
            'vocab_inp_size': vocab_inp_size,
            'vocab_tar_size': vocab_tar_size
        },
        {
            'encoder': encoder,
            'decoder': decoder
        }
    )


if __name__ == '__main__':
    train()
