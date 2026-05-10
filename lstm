import numpy as np
import tensorflow as tf

text = "hello world hello world hello world"

chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

seq_length = 5
X = []
y = []

for i in range(len(text) - seq_length):
    seq = text[i:i+seq_length]
    target = text[i+seq_length]

    X.append([char_to_idx[c] for c in seq])
    y.append(char_to_idx[target])

X = np.array(X)
y = np.array(y)


X = X.reshape((X.shape[0], X.shape[1], 1)) / len(chars)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

model.fit(X, y, epochs=50, verbose=1)

def generate(seed="hello", length=20):
    result = seed

    for _ in range(length):
        x_input = np.array([[char_to_idx[c] for c in result[-seq_length:]]])
        x_input = x_input.reshape((1, seq_length, 1)) / len(chars)

        pred = model.predict(x_input, verbose=0)
        next_char = idx_to_char[np.argmax(pred)]

        result += next_char

    return result

print(generate("hello", 30))
