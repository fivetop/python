from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import random
import io
import os
import pickle

"""
## Prepare the data
"""
# path = 'datasets/shakespeare.txt'
path = keras.utils.get_file(
    "shakespeare.txt", origin="https://norvig.com/ngrams/shakespeare.txt"
)
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")  # We remove newlines chars for nicer display
print("Corpus length:", len(text))

chars = sorted(list(set(text)))
print("Total chars:", len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

BASENAME = os.path.basename(path)
# save dicts for later gen
pickle.dump(char_indices, open(f"{BASENAME}-char_indices.pickle", "wb"))
pickle.dump(indices_char, open(f"{BASENAME}-indices_char.pickle", "wb"))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print("Number of sequences:", len(sentences))

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

"""
## Build the model: a single LSTM layer
"""

model = keras.Sequential(
    [
        keras.Input(shape=(maxlen, len(chars))),
        layers.LSTM(128),
        layers.Dense(len(chars), activation="softmax"),
    ]
)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule, clipnorm=1)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)
if os.path.exists("new-gen-2-epoch-1.hdf5"):
    model.load_weights("new-gen-2-epoch-1.hdf5")

"""
## Prepare the text sampling function
"""


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


"""
## Train the model
"""

epochs = 40
batch_size = 256

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

for epoch in range(epochs):
    with tf.device("/GPU:0"):
        model.fit(x, y, batch_size=batch_size, epochs=1, callbacks=[callback])
    model.save("new-gen-2-epoch-" + str(epoch) +".hdf5")
    print()
    print("Generating text after epoch: %d" % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 0.6, 1.0, 1.2]:
        print("...Diversity:", diversity)

        generated = ""
        sentence = text[start_index : start_index + maxlen]
        print('...Generating with seed: "' + sentence + '"')

        for i in range(240):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char

        print("...Generated: ", generated)
        print()