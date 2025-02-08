import tensorflow as tf
import matplotlib.pyplot as plt
from utils import load_fashion_mnist
import os  # For creating directories

# Train - Test set
(train_images, train_labels), (test_images, test_labels) = load_fashion_mnist()

# Normalize pixel values to be between 0 and 1
train_images  = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# One-hot encode the labels (important for categorical cross-entropy loss)
train_labels  = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# Create the "images" directory if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

def plot_and_save_samples(images, labels, num_samples=25, filename_prefix="fashion_sample"):
    """Plots and saves a grid of images with their labels."""

    num_rows = int(num_samples**0.5)  # Calculate rows and columns for square grid
    num_cols = int(num_samples**0.5)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5, 5))

    for i in range(num_samples):
        row = i // num_cols
        col = i % num_cols

        image = images[i]
        label = labels[i]  # One-Hot encoded labels

        # Convert one-hot encoded label back to integer
        predicted_label = tf.argmax(label).numpy()

        # Fashion MNIST 
        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        true_label_name = class_names[predicted_label]

        axes[row, col].imshow(image, cmap='gray')  # Display image
        axes[row, col].set_title(true_label_name) # Set title as class name
        axes[row, col].axis('off')  # Hide axis ticks and labels

    plt.tight_layout()
    plt.savefig(f"images/{filename_prefix}.png")  # Save the figure
    plt.show()


# Plot and save samples from the training set
plot_and_save_samples(train_images[:9], train_labels[:9], num_samples=9, filename_prefix="train_samples")

# Plot and save samples from the test set
plot_and_save_samples(test_images[:9], test_labels[:9], num_samples=9, filename_prefix="test_samples")