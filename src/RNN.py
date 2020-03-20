from __future__ import print_function

import os
import random
import sys

import numpy as np
from keras.callbacks import LambdaCallback
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop
from RNN_pre_processing import extract_characters

author = "shakespeare" # "eminem" # "shakespeare" # "spenser"
units = 200
maxlen = 40
step = 3
epochs = 100
batch_size = 128
fixed_starting_sentence = "shall i compare thee to a summer's day?\n"


if author == "eminem":
    file = open("data/eminem.txt", "r")
    text = file.read().lower()
else:
    text = extract_characters(author, True)

print('Text length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Split the text
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# LSTM model
print('Building model...')
model = Sequential()
model.add(LSTM(units, input_shape=(maxlen, len(chars)))) # , dropout = 0.3, recurrent_dropout = 0.3
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer
    # optimizer='Adam',
    # metrics=['accuracy']
)

# Invoked after every epoch to print text
def on_epoch_end(epoch, _):
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.25, 0.75, 1.5]:
        print('----- diversity:', diversity)

        generated = ''

        if fixed_starting_sentence:
            sentence = fixed_starting_sentence
        else:
            sentence = text[start_index: start_index + maxlen]

        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(800):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = get_index_from_proba(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()

        print()


# Get index from probability array
def get_index_from_proba(preds, temp=1.0):
    preds = np.log(np.asarray(preds).astype('float64')) / temp

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)


model_save_path = f"model_save/RNN_weights_{author}_{units}_{maxlen}_{step}_{epochs}_{batch_size}.h5"

if os.path.isfile(model_save_path):
    model.load_weights(model_save_path)

else:
    model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
    model.save_weights(model_save_path)

on_epoch_end(0, None)