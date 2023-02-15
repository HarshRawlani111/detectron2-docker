# This is a Streamlit App

import streamlit as st
import pickle
from prediction import Detector
import os
from PIL import Image
#import cv2
import numpy as np
# Save and Load Model
#pickle_in = open("classifier.pkl", "rb")
#classifier = pickle.load(pickle_in)
detector = Detector(filename="inputImage.jpg")

RENDER_FACTOR = 35

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# A custom function for predicting the values
def predictRoute():
    result = detector.inference('inputImage.jpg')
    #print(result)
    return jsonify(result)


def main():
    st.title("Detectron2 💵 ")

    st.markdown("""
    ## **Dataset Information : **
    *.  from images.**
    """,True)
    
    st.markdown("templates/home.html", unsafe_allow_html=True)
    filenamea = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    if filenamea is not None:
        filename = Image.open(filenamea)
        img_array = np.array(filename)
        #st.image(filename, caption="The caption{img_array.shape[0:2]}", use_column_width=True)

    result = ""
    





		
    if st.button("Predict"):
        result = detector.inference('inputImage.jpg')
        if result == 1:
            st.success("Image has required object")
        elif result ==2 :
            st.error("Image doesnt have required object")
        else:
            st.error("Image doesnt have required object") 

      

    if st.button("About"):
        st.markdown("""**Built with ❤️ by Harsh**""")


if __name__ == '__main__':
    main()