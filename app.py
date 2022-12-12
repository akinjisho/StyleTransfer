# -*- coding: utf-8 -*-
# Import required libraries

import cv2
import numpy as np
import streamlit as st
import requests
import os
import tensorflow as tf
# Home UI 

def main():

    st.set_page_config(layout="wide")

    font_css = """
        <style>
        button[data-baseweb="tab"] {
        font-size: 26px;
        }
        </style>
        """

    st.write(font_css, unsafe_allow_html=True)

    
    stleTransfer()

def loadImg():
    content_image_url = st.text_input('Content Image URL', 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg')
    style_image_url = st.text_input('Style Image URL', 'https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg')
    output_image_size = 384
    # The content image size can be arbitrary.
    content_img_size = (output_image_size, output_image_size)
    # The style prediction model was trained with image size 256 and it's the 
    # recommended image size for the style image (though, other sizes work as 
    # well but will lead to different results).
    style_img_size = (256, 256)  # Recommended to keep it at 256.

    #content_image = load_image(content_image_url, content_img_size)
    #style_image = load_image(style_image_url, style_img_size)
    content_image = Image.open(content_image_url)
    style_image = Image.open(style_image_url)
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

    return (content_image,style_image)

def stleTransfer():

    st.header("Image Cartoonification")

    (content_image,style_image) = loadImg()

    # Original Image
    st.subheader("Input Images")
    st.image(content_image,use_column_width=True)
    st.image(style_image,use_column_width=True)
    
    st.subheader("Cartoonized Image")

    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)

    
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))

    st.image(stylized_image)


if __name__ == "__main__":
    main()
