import tensorflow as tf
from tensorflow import keras
import numpy as np




model = keras.Sequential(
    [
        layers.Dense(30, activation="relu", name="layer1"),
        layers.Dense(10, activation="softmax", name="output"),
    ]
)


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


history = model.fit(trainx, trainy,
          batch_size=50, epochs=30,
          verbose=2,
          validation_data=(testx, testy))