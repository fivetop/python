from __future__ import print_function
import os
from os import path
# from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD, Adam
import numpy as np
import pickle

FILE_PATH = "datasets/sonnets.txt"
BASENAME = os.path.basename(FILE_PATH)
text = open(FILE_PATH).read().lower()
# .replace("\"", "")
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# save dicts for later gen
pickle.dump(char_indices, open(f"{BASENAME}-char_indices.pickle", "wb"))
pickle.dump(indices_char, open(f"{BASENAME}-indices_char.pickle", "wb"))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 25
step = 3
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

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(256, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

# optimizer = RMSprop(learning_rate=0.001)
# optimizer = SGD(learning_rate = 0.01)
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# training
if path.exists("shakespeare-gen-cp.hdf5"):
    model = load_model("shakespeare-gen-cp.hdf5")

# def sample(preds, temperature=1.0):
#     # helper function to sample an index from a probability array
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)
#
#
# def on_epoch_end(epoch, _):
#     # Function invoked at end of each epoch. Prints generated text.
#     print()
#     print('----- Generating text after Epoch: %d' % epoch)
#
#     start_index = random.randint(0, len(text) - maxlen - 1)
#     for diversity in [0.2, 0.5, 1.0, 1.2]:
#         print('----- diversity:', diversity)
#
#         generated = ''
#         sentence = text[start_index: start_index + maxlen]
#         generated += sentence
#         print('----- Generating with seed: "' + sentence + '"')
#         sys.stdout.write(generated)
#
#         for i in range(400):
#             x_pred = np.zeros((1, maxlen, len(chars)))
#             for t, char in enumerate(sentence):
#                 x_pred[0, t, char_indices[char]] = 1.
#
#             preds = model.predict(x_pred, verbose=0)[0]
#             next_index = sample(preds, diversity)
#             next_char = indices_char[next_index]
#
#             sentence = sentence[1:] + next_char
#
#             sys.stdout.write(next_char)
#             sys.stdout.flush()
#         print()
#
#
# print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
checkpoint = ModelCheckpoint("shakespeare-gen-cp.hdf5", monitor='loss', verbose=1,
                             save_best_only=True, mode='auto', period=1)

model.fit(x, y,
          batch_size=128,
          epochs=50,
          callbacks=[checkpoint]
          )
if not os.path.isdir("results"):
    os.mkdir("results")
model.save(f"results/shakespeare-model.h5")
print("Saved model to disk")
