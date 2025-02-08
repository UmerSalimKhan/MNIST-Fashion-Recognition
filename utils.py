import tensorflow as tf

def load_fashion_mnist():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)