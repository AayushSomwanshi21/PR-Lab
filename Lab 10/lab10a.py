import numpy as np
import keras
from keras.utils import to_categorical
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

data = load_digits()
x, y = data.data, data.target

x = x.reshape(-1, 8, 8, 1).astype('float32') / 16.0
y = to_categorical(y, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=8, kernel_size=(
        3, 3), activation='relu', input_shape=(8, 8, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, verbose=1,
          validation_data=(x_test, y_test))

_, test_acc = model.evaluate(x_test, y_test)
print(f'Accuracy: {test_acc}')
