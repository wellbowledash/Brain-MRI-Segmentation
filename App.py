import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Layer
import cv2

plt.style.use("ggplot")

@tf.keras.utils.register_keras_serializable()
def dice_coefficients(y_true, y_pred, smooth=1e-5):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

@tf.keras.utils.register_keras_serializable()
def dice_coefficients_loss(y_true, y_pred, smooth=1e-5):
    return 1 -dice_coefficients(y_true, y_pred, smooth)

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_coefficients_loss(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def iou(y_true, y_pred, smooth=1e-5):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection

    iou = (intersection + smooth) / (union + smooth)
    
    return K.mean(iou)

@tf.keras.utils.register_keras_serializable()    
class ResizeLayer(Layer):
    def __init__(self, target_size, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size)
    
    def get_config(self):
        config = super(ResizeLayer, self).get_config()
        config.update({"target_size": self.target_size})
        return config


st.title("Brain MRI Segmentation App")
model = load_model('unet_brain_mri_seg3.keras', custom_objects={
    'dice_coefficients': dice_coefficients,
    'dice_coefficients_loss': dice_coefficients_loss,
    'combined_loss': combined_loss,
    'iou': iou,
    'ResizeLayer': ResizeLayer
    
})

im_height = 256
im_width = 256

files = st.file_uploader("Upload Brain MRI images", type=[
                            "csv", "png", "jpg", "tif"], accept_multiple_files=True)

if files:
    # Iterate through each uploaded file
    for i, file in enumerate(files):
        if i >= 10:  # Limit to 10 images
            break

        st.header(f"Image {i+1}")

        # Create two columns
        col1, col2 = st.columns(2)

        # Display original image in the first column
        with col1:
            st.image(file, caption="Original Image")

        # Process the image
        content = file.getvalue()
        image = np.asarray(bytearray(content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        img2 = cv2.resize(image, (im_height, im_width))
        img3 = img2 / 255
        img4 = img3[np.newaxis, :, :, :]

        # Display the predict button for each image
        if st.button(f"Predict Output {i+1}"):
            pred_img = model.predict(img4)
            pred_img = (pred_img > 0.5).astype(np.uint8) * 255  # Binary threshold to make tumor regions white
            pred_img = pred_img[0, :, :, 0]  # Remove batch and channel dimensions

            # Display predicted mask in the second column
            with col2:
                st.image(pred_img, caption="Predicted Mask", clamp=True)