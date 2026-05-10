import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import SimpleRNN,Dense
from tensorflow.keras.models import Sequential

def generate_sequence(n):
    x = np.linspace(0, 50, n)
    y = np.sin(x)
    return y
x = []
y = []

sequence = generate_sequence(100)
seq_len=10

for i in range(len(sequence) - seq_len):
    x.append(sequence[i:i+seq_len])
    y.append(sequence[i+seq_len])

x, y = np.array(x), np.array(y)
x = x.reshape((x.shape[0], x.shape[1], 1))
model = Sequential([
    SimpleRNN(10, activation='relu', input_shape=(seq_len,1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=200, verbose=1)

prediction = model.predict(x)
print(f"Excepted:{y[:5]}")
print(f"Predicted:{prediction[:5].flatten()}")
