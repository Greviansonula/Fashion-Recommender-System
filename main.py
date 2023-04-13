import streamlit as st 
import os
from PIL import Image
import pickle
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

features_list = np.array(pickle.load(open('artifacts/embeddings.pkl', 'rb')))
filenames = np.array(pickle.load(open('artifacts/filenames.pkl', 'rb')))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        print(e)
        return 0
    
def feature_extract(img_path, model):
    img = image.load_img('data/myntradataset/images/60000.jpg', target_size=(224, 224))
    img_array = image.img_to_array(img) # (224, 224, 3)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        features = feature_extract(os.path.join("uploads", uploaded_file.name), model)
        st.text(features)
    else:
        st.header("Some error occurred during upload")
    
