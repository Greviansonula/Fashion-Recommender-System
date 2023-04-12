import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# print(model.summary())

def extract_features(img,  model):
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img) # (224, 224, 3)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

filenames = []
for file in tqdm(os.listdir("data/images")):
    filenames.append(os.path.join('data', 'images', file))

feature_list = []

for idx, file in tqdm(enumerate(filenames)):
    feature_list.append(extract_features(file, model))
    if idx == 2000:
        break


pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

print(np.array(feature_list).shape)