import tensorflow as tf


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0


X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)


optimizers = {
    "SGD": tf.keras.optimizers.SGD(),
    "Adam": tf.keras.optimizers.Adam(),
    "RMSProp": tf.keras.optimizers.RMSprop(),
    "Adadelta": tf.keras.optimizers.Adadelta(),
    "Adagrad": tf.keras.optimizers.Adagrad(),
    "Nadam": tf.keras.optimizers.Nadam(),
    "Adamax": tf.keras.optimizers.Adamax()
}


for name, optimizer in optimizers.items():

    print("\nTraining using", name)

    model = tf.keras.Sequential([
          tf.keras.layers.Dense(100,input_dim=X_train.shape[1],activation="relu"),
          tf.keras.layers.Dense(50,activation="relu"),
          tf.keras.layers.Dense(25,activation="relu"),
          tf.keras.layers.Dense(10,activation="softmax")
    ])

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        epochs=3,
        validation_split=0.2
    )

    loss, accuracy = model.evaluate(X_test, y_test)

    print("Loss:", loss)
    print("Accuracy:", accuracy * 100, "%")
