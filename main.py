import streamlit as st
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

label_name = [
    'Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
    'Cherry healthy', 'Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 
    'Corn Northern Leaf Blight', 'Corn healthy', 'Grape Black rot', 'Grape Esca', 
    'Grape Leaf blight', 'Grape healthy', 'Peach Bacterial spot', 'Peach healthy', 
    'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight', 
    'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy', 
    'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 
    'Tomato Septoria leaf spot', 'Tomato Spider mites', 'Tomato Target Spot', 
    'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy'
]

st.title("Leaf Disease Detection")
st.markdown("### Upload an image of a leaf to detect its disease.")

st.write("""
The leaf disease detection model is built using deep learning techniques, and it uses transfer learning to leverage the pre-trained knowledge of a base model.
Please input only leaf images of Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, and Tomato.
""")

model = load_model('Training/model/Leaf Deases(96,88).h5')

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.read()
        img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
        if img is None:
            raise ValueError("The file is not a valid image.")
        
        img_resized = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150))
        normalized_image = np.expand_dims(img_resized, axis=0) / 255.0
        predictions = model.predict(normalized_image)
        st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
        
        max_index = np.argmax(predictions)
        confidence = predictions[0][max_index] * 100
        if confidence >= 80:
            st.success(f"Prediction: {label_name[max_index]} ({confidence:.2f}%)")
        else:
            st.warning("Unable to confidently identify the disease. Try another image.")
    except Exception as e:
        st.error(f"An error occurred: {e}")