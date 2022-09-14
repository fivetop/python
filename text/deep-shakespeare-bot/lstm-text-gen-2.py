import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import RMSprop


sequence_length = 50
BATCH_SIZE = 128
EPOCHS = 45
# dataset file path
FILE_PATH = "datasets/shakespeare.txt"
BASENAME = os.path.basename(FILE_PATH)
# read the data
text = open(FILE_PATH).read().lower().replace("\"", "")

# print stats
n_chars = len(text)
chars = sorted(list(set(text)))
vocab = ''.join(sorted(set(text)))
print("unique characters: ", vocab)
n_unique_chars = len(chars)
print("number of characters: ", n_chars)
print("number of unique chars", n_unique_chars)

# dict that converts chars to ints
char2int = {c: i for i, c in enumerate(chars)}
# dict that converts ints back into chars
int2char = {i: c for i, c in enumerate(chars)}

# save dicts for later gen
pickle.dump(char2int, open(f"{BASENAME}-char2int.pickle", "wb"))
pickle.dump(int2char, open(f"{BASENAME}-int2char.pickle", "wb"))

# convert all text into ints
encoded_text = np.array([char2int[c] for c in text])

# construct tf.data.Dataset object for efficient dataset handling
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

# build sequences by batching
sequences = char_dataset.batch(2 * sequence_length + 1, drop_remainder=True)


def split_sample(sample):
    ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
    for i in range(1, (len(sample - 1)) // 2):
        input_ = sample[i: i + sequence_length]
        target = sample[i + sequence_length]
        # extend dataset with samples by concatenate() method
        other_ds = tf.data.Dataset.from_tensors((input_, target))
        ds = ds.concatenate(other_ds)
    return ds


# prepare inputs and targets
dataset = sequences.flat_map(split_sample)


# one-hot encode the inputs and targets
def one_hot_samples(input_, target):
    return tf.one_hot(input_, n_unique_chars), tf.one_hot(target, n_unique_chars)


# one-hot encode each sample in dataset
dataset = dataset.map(one_hot_samples)

# repeat, shuffle, batch dataset
ds = dataset.repeat().shuffle(1024).batch(BATCH_SIZE, drop_remainder=True)

# building model
model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax"),
])
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# train model
if not os.path.isdir("results"):
    os.mkdir("results")
model.fit(ds, steps_per_epoch=(len(encoded_text) - sequence_length) // BATCH_SIZE, epochs=EPOCHS)
model.save(f"results/{BASENAME}-{sequence_length}-2.h5")


