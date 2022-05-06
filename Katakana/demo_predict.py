import heapq
import numpy as np
import tensorflow as tf
from tensorflow import keras
reconstructed_model = keras.models.load_model("katakana-model.h5")
test_images = np.load("katakana_test_images.npz")['arr_0']
test_labels = np.load("katakana_test_labels.npz")['arr_0']
prds = reconstructed_model.predict(test_images)

test_loss, test_acc = reconstructed_model.evaluate(test_images, test_labels)
print("################### Model Summary:\n")
reconstructed_model.summary()

print("################### Test Accuracy:\n")
print(test_acc)

print(f'################### Predictions (prds, labels):\n')
[print(f'prd: {z[0].argmax()} vs label: {z[1]}\n') for z in zip(prds, test_labels)]
