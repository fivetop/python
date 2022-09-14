import numpy as np
import pickle
import random
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
import os

maxlen = 50
sequence_length = 50
# dataset file path
FILE_PATH = "datasets/sonnets.txt"
WRITE_PATH = "gen-queue.txt"
BASENAME = os.path.basename(FILE_PATH)
# load dataset text
text = open(FILE_PATH, encoding="utf-8").read()
# remove caps and quotes, comment this code if you want uppercase characters as well
text = text.lower()
# .replace("\"", "")
chars = sorted(list(set(text)))

# load vocab dictionaries
char_indices = pickle.load(open(f"{BASENAME}-char_indices.pickle", "rb"))
indices_char = pickle.load(open(f"{BASENAME}-indices_char.pickle", "rb"))

# building the model
model = Sequential()
model.add(LSTM(256, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

# load the optimal weights
# model.load_weights(f"results/shakespeare-model.h5")
model.load_weights("shakespeare-gen-cp.hdf5")


# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


with open(WRITE_PATH, "w") as queue:
    # write 100000 examples to text
    print("writing generated text to gen-queue.txt")
    for n in range(10000):
        # random start index
        start_index = random.randint(0, len(text) - maxlen - 1)

        # temperature values
        # for diversity in [0.2, 0.5, 1.0, 1.2]:

        # generate seed
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        # print("Seed: " + "\n" + sentence)

        # generate
        for i in range(230):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            # next_index = sample(preds, diversity)
            next_index = np.argmax(preds)
            next_char = indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char
        # print("Generated text: " + "\n" + generated)
        queue.write(generated + "%")
print("finished writing")
