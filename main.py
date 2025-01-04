import streamlit as st
from PIL import Image
from utils import *

# Sidebar
st.sidebar.header("User Entries")
style_weight = st.sidebar.number_input("Enter Style Weight:",value=1e6)
content_weight = st.sidebar.number_input("Enter Content Weight:", value = 1)
epochs = st.sidebar.number_input("Enter Epoch Count:", value = 1000)
show_image_button = st.sidebar.button("Show Image in Output")

# Main Layout
st.title("Neural Style Transfer")

col1, col2 = st.columns(2)

with col1:
    st.header("Image Input and Display")
    uploaded_file1 = st.file_uploader("Choose the first image file", type=["jpg", "jpeg", "png"], key="uploader1")
    uploaded_file2 = st.file_uploader("Choose the second image file", type=["jpg", "jpeg", "png"], key="uploader2")

    if uploaded_file1:
        image1 = Image.open(uploaded_file1)
        st.image(image1, caption="First Uploaded Image", use_container_width=True)

    if uploaded_file2:
        image2 = Image.open(uploaded_file2)
        st.image(image2, caption="Second Uploaded Image", use_container_width=True)
with col2:
    st.header("Output Image")
    if uploaded_file1 and uploaded_file2 and show_image_button:
        img = model_train(image1,image2,style_weight,content_weight,epochs)
        #image = Image.open(uploaded_file)
        st.image(img, caption="Output Image", use_container_width=True)