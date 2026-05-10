import tensorflow as tf
import numpy as np

(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
print("Training Shape:", X_train_full.shape)
print("Test Shape:", X_test.shape)

X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_train = X_train_full[:-5000]
X_val = X_train_full[-5000:]
y_train = y_train_full[:-5000]
y_val = y_train_full[-5000:]

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("\nUpdated Training Shape:", X_train.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,kernel_size=3,padding="same",activation="relu",input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64,kernel_size=3,padding="same",activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10,activation="softmax")

])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.summary()
model.fit(X_train,y_train,epochs=5,validation_data=(X_val, y_val))

print("\nTest Accuracy:")
print(model.evaluate(X_test, y_test)[1])
