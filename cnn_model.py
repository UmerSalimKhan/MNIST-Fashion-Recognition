# cnn_model.py
import tensorflow as tf

def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):  # Added input_shape parameter
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),  # Convolutional layer 1
        tf.keras.layers.MaxPooling2D((2, 2)),  # Max pooling layer 1
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Convolutional layer 2
        tf.keras.layers.MaxPooling2D((2, 2)),  # Max pooling layer 2
        tf.keras.layers.Flatten(),  # Flatten the output
        tf.keras.layers.Dense(128, activation='relu'), # Dense Layer
        tf.keras.layers.Dropout(0.2),  # Dropout for regularization
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer
    ])
    return model