import numpy as np
import keras
from keras import datasets
from keras.utils import to_categorical

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model.fit(x_train, y_train, epochs=5, verbose=1,
          validation_data=(x_test, y_test))
_, test_acc = model.evaluate(x_test, y_test)
print(f'Accuracy: {test_acc}')
