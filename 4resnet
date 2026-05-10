import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train[:5000]
y_train = y_train[:5000]

X_test = X_test[:1000]
y_test = y_test[:1000]


X_train = X_train / 255.0
X_test = X_test / 255.0


X_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_train[..., tf.newaxis]))
X_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_test[..., tf.newaxis]))


X_train = tf.image.resize(X_train, (96,96))
X_test = tf.image.resize(X_test, (96,96))


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(96,96,3))

base_model.trainable = False


model = Sequential([base_model, GlobalAveragePooling2D(), Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

loss, acc = model.evaluate(X_test, y_test)
print("Before Fine-Tuning Accuracy:", acc)

base_model.trainable = True

model.fit(X_train, y_train, epochs=10)

# Accuracy after fine-tuning
loss, acc = model.evaluate(X_test, y_test)
print("After Fine-Tuning Accuracy:", acc)
