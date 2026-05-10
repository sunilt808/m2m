!pip install tensorflow-model-optimization  #run this in one cell and then restart the session from the runtime and then run next cell
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

baseline_acc = model.evaluate(x_test, y_test, verbose=0)[1]
print("Baseline Accuracy:", baseline_acc)

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, tfmot.sparsity.keras.PolynomialDecay(0.2, 0.8, 0, 1000))
pruned_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
pruning_callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
pruned_model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), callbacks=pruning_callbacks)
pruned_acc = pruned_model.evaluate(x_test, y_test, verbose=0)[1]
print("Pruned Accuracy:", pruned_acc)

converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()
print("Quantized Model Size:",len(quantized_model) / 1024, "KB")
