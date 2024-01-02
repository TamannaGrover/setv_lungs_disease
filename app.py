import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
import os

# Path to the trained model
model_path = os.path.join('Model','mymodel.h5')

# Load the trained model
model = load_model(model_path)

# Mapping indices to lung conditions
results = {
    0: 'Normal',
    1: 'Cancer',
    2: 'Covid',
    3: 'Fibrosis'
}

# Function to process the image and predict
def predict_image(image):
    width, height = 224, 224
    image = image.resize((width, height))
    image_array = np.expand_dims(np.array(image), axis=0)
    image_array = image_array / 255.0

    predictions = model.predict(image_array)[0]
    predicted_class = np.argmax(predictions)
    
    return results[predicted_class], predictions[predicted_class]

# Set up the main app
def main():
    st.title("Lung Condition Classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Predict and display the results
        label, prob = predict_image(image)
        st.write(f"Predicted class: {label}")
        st.write(f"Probability: {prob:.4f}")

if __name__ == "__main__":
    main()
