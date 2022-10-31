import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import UnivariateSpline
def LookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))

st.markdown('<p style="text-align: center; font-size: 50px;">Image Editing Application</p>',unsafe_allow_html=True)
with st.sidebar.expander("About"):
    st.write("""
           A simple app to convert photos of your choice into various exciting filters, or find the colors that comprise an image.\n \n Created by Saumya Joshi as part of IVP mini project 2022-23.
        """)
uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])
filter = st.sidebar.radio("Convert your Photo to:", ['Grayscale', 'Pencil Sketch', 'HDR', 'Warm Filter', 'Cool Filter', 'Blur Effect'])  # Add the filter in the sidebar

rotate_image = st.sidebar.slider('Rotate to (in degrees):', 0, 360)

if uploaded_file is not None:
   image = Image.open(uploaded_file)
   col1, col2 = st.columns([0.5, 0.5])
   with col1:
        st.markdown('<p style="text-align: center;">Original Image</p>',unsafe_allow_html=True)
        st.image(image, width=300)
   with col2:
       converted_img = np.array(image.convert('RGB'))
       (h, w) = converted_img.shape[:2]
       (cX, cY) = (w // 2, h // 2)
       M = cv2.getRotationMatrix2D((cX, cY), rotate_image, 1.0)
       rotated = cv2.warpAffine(converted_img, M, (w, h))
       st.markdown(f'<p style="text-align: center;">{filter} Image </p>', unsafe_allow_html=True)
       if filter == 'Grayscale':

           gray_scale = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
           st.image(gray_scale, width=300)
       elif filter == "HDR":

           hdr = cv2.detailEnhance(rotated, sigma_s=12, sigma_r=0.15)
           st.image(hdr, width=300)
       elif filter == 'Pencil Sketch':

           gray_scale = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
           inv_gray = 255 - gray_scale
           slider = st.sidebar.slider('Adjust the intensity', 25, 255, 125, step=2)
           blur_image = cv2.GaussianBlur(inv_gray, (slider, slider), 0, 0)
           sketch = cv2.divide(gray_scale, 255 - blur_image, scale=256)
           st.image(sketch, width=300)
       elif filter == 'Blur Effect':

           slider = st.sidebar.slider('Adjust the intensity', 5, 81, 33, step=2)
           rotated = cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)
           blur_image = cv2.GaussianBlur(rotated, (slider, slider), 0, 0)
           st.image(blur_image, channels='BGR', width=300)
       elif filter == 'Warm Filter':
           rotated = np.array(image.convert('RGB'))
           increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
           decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
           blue_channel, green_channel, red_channel = cv2.split(rotated)
           red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
           blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
           warm_effect = cv2.merge((blue_channel, green_channel, red_channel))
           st.image(warm_effect, channels='BGR', width=300)
       elif filter == 'Cool Filter':
           increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
           decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
           blue_channel, green_channel, red_channel = cv2.split(rotated)
           red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
           blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
           cool_effect = cv2.merge((blue_channel, green_channel, red_channel))
           st.image(cool_effect, channels='BGR', width=300)
       else:
           st.image(image, width=300)
