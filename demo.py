import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import cv2

TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
IMAGE_SHAPE = (321, 321)

df = pd.read_csv(LABEL_MAP_URL)
label_map = dict(zip(df.id, df.name))

class LandmarkClassifier(tf.keras.Model):
    def __init__(self, model_url, output_key):
        super(LandmarkClassifier, self).__init__()
        self.keras_layer = hub.KerasLayer(model_url, input_shape=IMAGE_SHAPE + (3,), output_key=output_key)

    def call(self, inputs):
        return self.keras_layer(inputs)

def classify_img(RGBimg):
    RGBimg = np.array(RGBimg) / 255
    RGBimg = np.reshape(RGBimg, (1, 321, 321, 3))
    prediction = model.predict(RGBimg)
    return label_map[np.argmax(prediction)]

def main():
    st.title("Statueist")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        RGBimg = cv2.resize(RGBimg, (321, 321))
        result = classify_img(RGBimg)
        
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("Predicted Landmark:", result)

if __name__ == "__main__":
    model = LandmarkClassifier(TF_MODEL_URL, output_key="predictions:logits")
    main()
