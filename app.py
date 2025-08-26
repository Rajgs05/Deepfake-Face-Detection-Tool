import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# --- Define all custom components required to load the Keras model ---

IMG_HEIGHT, IMG_WIDTH = 224, 224

def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))

def compute_capsule_length(x):
    return K.sqrt(K.sum(K.square(x), -1))
    
@tf.keras.utils.register_keras_serializable()
class CapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule, self.dim_capsule, self.input_dim_capsule], initializer='glorot_uniform', name='W')
        self.built = True
    def call(self, inputs, training=None):
        u_hat = tf.einsum('ijkm,bjm->bijk', self.W, inputs)
        b = tf.zeros(shape=[tf.shape(u_hat)[0], self.num_capsule, self.input_num_capsule])
        outputs = None
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            s = tf.einsum('bij,bijk->bik', c, u_hat)
            outputs = squash(s)
            if i < self.routings - 1:
                agreement = tf.einsum('bik,bijk->bij', outputs, u_hat)
                b += agreement
        return outputs
    def get_config(self):
        config = super().get_config()
        config.update({"num_capsule": self.num_capsule, "dim_capsule": self.dim_capsule, "routings": self.routings})
        return config

# --- Streamlit App Interface ---

@st.cache_resource
def load_trained_model():
    custom_objects = {
        'CapsuleLayer': CapsuleLayer, 
        'margin_loss': margin_loss, 
        'squash': squash, 
        'compute_capsule_length': compute_capsule_length
    }
    model = load_model('best_hybrid_model.h5', custom_objects=custom_objects)
    return model

model = load_trained_model()

class_names = {0: 'FAKE', 1: 'REAL'} 

st.set_page_config(layout="centered", page_title="Deepfake Detector")
st.title("🤖 Deepfake Image Detector")
st.write("Upload an image and the AI will analyze it to determine if it is a real photograph or a synthetically generated deepfake.")

uploaded_file = st.file_uploader("Choose an image to analyze...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('AI is thinking...'):
        prediction_lengths = model.predict(img_array)[0]
        predicted_class_index = np.argmax(prediction_lengths)
        predicted_class_name = class_names[predicted_class_index]
        confidence = prediction_lengths[predicted_class_index]

    with col2:
        st.write("")
        st.write("")
        st.write("")
        if predicted_class_name == 'FAKE':
            st.error(f"### Prediction: This image is a FAKE.")
        else:
            st.success(f"### Prediction: This image is REAL.")
        
        st.write(f"**Confidence (Capsule Length):** `{confidence:.4f}`")
        st.info("The 'Confidence' is the length of the winning capsule vector. A value closer to 1.0 indicates higher model confidence.", icon="ℹ️")