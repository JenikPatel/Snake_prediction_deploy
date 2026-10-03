from fastai.vision.all import *
import json
import pathlib, pathlib._local
from fastcore.all import *
import streamlit as st
import sys

@st.cache_resource
def load_model():
    return load_learner('export_2.pkl')

@st.cache_data
def load_info():
    with open('snake_info.json', encoding = 'utf-8') as f:
        return json.load(f)

if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath
    pathlib._local.PosixPath = pathlib.WindowsPath

uploaded_image = st.file_uploader("Upload Snake image", type = ["jpg", "jpeg", "png"])

if uploaded_image is not None:
    img = PILImage.create(uploaded_image.getvalue())
    st.image(img, caption = "Your uploaded image")
    learn_inf = load_model()
    info = load_info()
    pred, pred_indx, prob = learn_inf.predict(img)
    st.write(f"#### **Prediction:** {pred}; **Probability:** {prob[pred_indx]:.4f}")
    card = info[pred]

    st.write("#### **Common Name:**",card['common_name'])
    st.caption("##### Also known as: " + ",".join(card['local_names']))
    st.write("**Venomous:** ", card['venomous'])
    st.write("**Venom type:** ", card['venom_type'])
    st.write("**Danger:** ", card['danger'])
    st.write("**Introduction:** ", card['introduction'])
    st.write("**Range:** ", card['range'])
    st.write("**Habitat:** ", card['habitat'])
    st.write("**Diet:** ", card['diet'])
    st.write("**Length:** ", card['length'])
    st.write("**Activity:** ", card['activity'])
    st.write("**Reproduction:** ", card['reproduction'])
    st.write("**Method to recognize:** ", card['how_to_recognise'])
