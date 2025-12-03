import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Clean Keras imports
try:
    from tensorflow.keras.models import load_model
except Exception:
    from keras.models import load_model

try:
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2,
        preprocess_input,
        decode_predictions
    )
except Exception:
    # Fallback to standalone keras if tensorflow.keras is unavailable or not resolved by the environment/linter
    from keras.applications.mobilenet_v2 import (
        MobileNetV2,
        preprocess_input,
        decode_predictions
    )

# -----------------------------
# MobileNetV2 ImageNet function
# -----------------------------
def mobilenetv2_imagenet():
    st.title("Image Classification with MobileNetV2")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("Classifying...")

        # Load MobileNetV2
        model = MobileNetV2(weights="imagenet")

        # Preprocess
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        predictions = model.predict(img_array)
        decoded = decode_predictions(predictions, top=1)[0]

        for (_, label, score) in decoded:
            st.write(f"{label}: {score * 100:.2f}%")

# -----------------------------
# CIFAR-10 function
# -----------------------------
def cifar10_classification():
    st.title("CIFAR-10 Image Classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("Classifying...")

        # Load CIFAR-10 model
        model = load_model("cifar10_model.h5")

        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        # Preprocess to 32x32x3
        img = image.resize((32, 32))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence * 100:.2f}%")

# -----------------------------
# MAIN
# -----------------------------
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Choose Model", ("CIFAR-10", "MobileNetV2 (ImageNet)"))

    if choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10":
        cifar10_classification()

if __name__ == "__main__":
    main()
