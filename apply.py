import tensorflow as tf
import pickle
import os
from tensorflow import keras
from config import EMBEDDING_DIM, MODEL_PATH, UNITS, BATCH_SIZE
from train import preprocess_english, Encoder, Decoder


def apply(sentences):
    with open(os.path.join(MODEL_PATH, 'input_lang_tokenizer.pickle'), 'rb') as handle:
        input_lang_tokenizer = pickle.load(handle)
    with open(os.path.join(MODEL_PATH, 'target_lang_tokenizer.pickle'), 'rb') as handle:
        target_lang_tokenizer = pickle.load(handle)
    with open(os.path.join(MODEL_PATH, 'max_input_length.pickle'), 'rb') as handle:
        max_input_length = pickle.load(handle)
    with open(os.path.join(MODEL_PATH, 'max_target_length.pickle'), 'rb') as handle:
        max_target_length = pickle.load(handle)
    with open(os.path.join(MODEL_PATH, 'vocab_inp_size.pickle'), 'rb') as handle:
        vocab_inp_size = pickle.load(handle)
    with open(os.path.join(MODEL_PATH, 'vocab_tar_size.pickle'), 'rb') as handle:
        vocab_tar_size = pickle.load(handle)

    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)

    encoder.load_weights(os.path.join(MODEL_PATH, 'encoder.ckpt'))
    decoder.load_weights(os.path.join(MODEL_PATH, 'decoder.ckpt'))

    print('Load model from %s successfully.' % MODEL_PATH)

    def translate(sentence):
        sentence = preprocess_english(sentence)

        inputs = [input_lang_tokenizer.word_index[i]
                  for i in sentence.split(' ')]
        inputs = keras.preprocessing.sequence.pad_sequences([inputs],
                                                            maxlen=max_input_length,
                                                            padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, UNITS))]
        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(
            [target_lang_tokenizer.word_index['<start>']], 0)

        for _ in range(max_target_length):
            predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                                 dec_hidden,
                                                                 enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1, ))

            predicted_id = tf.argmax(predictions[0]).numpy()

            if target_lang_tokenizer.index_word[predicted_id] == '<end>':
                return result

            result += target_lang_tokenizer.index_word[predicted_id]

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result

    pairs = {}

    for sentence in sentences:
        try:
            pairs.update({sentence: translate(sentence)})
        except KeyError as e:
            pairs.update({sentence: repr(e)})

    return pairs
