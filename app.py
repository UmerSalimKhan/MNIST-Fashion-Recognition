import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the saved CNN model
model = tf.keras.models.load_model("cnn_fashion_mnist_model.keras")

# Fashion MNIST class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

st.title("Fashion MNIST Classifier")

# File uploader
uploaded_file = st.file_uploader("Choose a Fashion MNIST image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_container_width =True)

    # Preprocess the image
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image)  # Convert to NumPy array
    image_array = image_array.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
    image_array = image_array.astype("float32") / 255.0  # Normalize

    if st.button("Predict"):
        # Make prediction
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions)
        predicted_label = class_names[predicted_class]

        st.write(f"## Predicted Label: {predicted_label}")

        # Display probabilities for each class (optional)
        st.write("### Class Probabilities:")
        for i, class_name in enumerate(class_names):
            probability = predictions[0][i] * 100
            st.write(f"- {class_name}: {probability:.2f}%")