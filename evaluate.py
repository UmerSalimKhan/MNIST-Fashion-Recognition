# evaluate.py
import tensorflow as tf
from utils import load_fashion_mnist # If you put the function in utils.py

(train_images, train_labels), (test_images, test_labels) = load_fashion_mnist()

# Normalize and one-hot encode (same as in train.py)
test_images = test_images.astype('float32') / 255.0
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)


model = tf.keras.models.load_model('cnn_fashion_mnist_model.keras') # Load your saved model
loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")