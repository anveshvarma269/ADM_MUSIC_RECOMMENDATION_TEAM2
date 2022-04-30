import streamlit as st
import IPython.display as ipd
#import cosine_similarity
from sklearn import preprocessing


st.header("Music Recommendation System")
st.write("Upload any music and get similar music")

uploaded_file = st.file_uploader("Choose an audio file...")