# train.py
import tensorflow as tf
from utils import load_fashion_mnist 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # Import for data splitting
from tensorflow.keras.preprocessing.image import ImageDataGenerator # For data augmentation
from tensorflow.keras.callbacks import EarlyStopping
from cnn_model import create_cnn_model
from tensorflow.keras.utils import plot_model


# Train - Test set
(train_images, train_labels), (test_images, test_labels) = load_fashion_mnist()

# Normalize pixel values to be between 0 and 1
train_images  = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# One-hot encode the labels (important for categorical cross-entropy loss)
train_labels  = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# print(train_images.shape) # (60000, 28, 28)
# print(train_labels.shape) # (60000, 10)

# Split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=42  # 10% for validation
)

# Reshape train_images to add the channel dimension
train_images = train_images.reshape(-1, 28, 28, 1)  # Add channel dimension (1 for grayscale)
val_images = val_images.reshape(-1, 28, 28, 1)      # Reshape the validation images as well
test_images = test_images.reshape(-1, 28, 28, 1) #Reshape the test images as well

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,  # Rotate images up to 10 degrees
    width_shift_range=0.1,  # Shift width up to 10%
    height_shift_range=0.1,  # Shift height up to 10%
    horizontal_flip=True  # Flip images horizontally
)

datagen.fit(train_images) # Fit the datagen to your training data

# Model Architechture
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)), # Flatten the images
#     # tf.keras.layers.Dense(256, activation='relu'), # Layer 1
#     # tf.keras.layers.BatchNormalization(), # Batch Normalization
#     # tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(128, activation='relu'), # Layer 2
#     tf.keras.layers.BatchNormalization(), # Batch Normalization
#     # tf.keras.layers.Dropout(0.1),
#     tf.keras.layers.Dense(64, activation='relu'), # Layer 3
#     tf.keras.layers.BatchNormalization(), # Batch Normalization
#     # tf.keras.layers.Dropout(0.1),
#     tf.keras.layers.Dense(10, activation='softmax') # 10 output classes (fashion categories)
# ])

# Create the CNN model
model = create_cnn_model()

plot_model(model, to_file='cnn_model_architecture.png', show_shapes=True, show_dtype=True, show_layer_names=True) # Save as PNG

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary() # Print the model architecture

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) # Patience = no. of epochs to wait

# Training model
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    epochs=15, 
    validation_data=(val_images, val_labels),
    callbacks=[early_stopping]) 

# Saving model
model.save('cnn_fashion_mnist_model.keras')

# Plot the training history 
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('images/cnn_history_simpler_model.png')
plt.show()