import streamlit as st
from audio_feature.audio_featurizer import audio_process, spectrogram_plot
from models.load_model import model_loader
import numpy as np
import pickle
from pydub import AudioSegment
import os
import IPython.display as ipd
import pandas as pd
import scipy as sc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import manifold
from sklearn import preprocessing
from pydub import AudioSegment
import os
from scipy.io import wavfile

model, encoding = model_loader("Saved_model.sav", "Encodings.sav")

st.sidebar.markdown(
    """<h1 style='text-align: center;color:  #0e76a8;'><a style='text-align: center;color:  #0e76a8;' href="https://www.linkedin.com/in/anveshvarma26/" target="_blank">Anvesh Varma Vatsavai</a></h1>""",
    unsafe_allow_html=True)
st.sidebar.markdown(
    """<h1 style='text-align: center;color:  #0e76a8;'><a style='text-align: center;color:  #0e76a8;' href = "https://www.linkedin.com/in/vishnutejapakalapati/" target="_blank">Vishnu Teja Pakalapti</a></h1>""",
    unsafe_allow_html=True)
st.image('Music-Recommendation.jpeg', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
# st.sidebar.markdown("""<h1 style='text-align: center;color: black;' ><a style='text-align: center;color: black;'href="https://github.com/anveshvarma26/TEATH" target="_blank">Github Source Code</a></h1>""", unsafe_allow_html=True)

st.sidebar.markdown(
    """<style>body {background-color: #2C3454; background-image: ('Music-Recommendation.jpeg'); color:white;}</style><body></body>""",
    unsafe_allow_html=True)
st.markdown(
    """<h1 style='text-align: center; color: white;font-size:60px;margin-top:-50px;'>Music Recommender and Genre Identifier</h1><h1 style='text-align: center; color: white;font-size:20px;margin-top:-30px;'></h1>""",
    unsafe_allow_html=True)

#radio = st.sidebar.radio("Select format of audio file", options=['mp3', 'wav'])



check = st.sidebar.checkbox('Start the System')

if check:



#else:
    radio = st.sidebar.radio("Select format of audio file", options=['wav'])

    if radio == 'wav':

        file = st.sidebar.file_uploader("Upload any Audio To know the Genre", type=["wav"])

        if file is not None:
            st.markdown(
                """<h1 style='color:yellow;'>Audio : </h1>""",
                unsafe_allow_html=True)
            st.audio(file)

            rad = st.sidebar.radio("Choose Options", options=["Predict"])

            # rad = st.sidebar.checkbox(label="Do You want to see the spectrogram ?")
            if rad == "Predict":
                if st.button("Classify Audio"):
                    uploaded_audio = audio_process(file)

                    predictions = model.predict(uploaded_audio)

                    targets = encoding.inverse_transform(np.array(predictions).reshape(1, -1))
                    #
                    # st.write(targets[0][0])
                    #
                    # st.success(targets[0][0])

                    st.markdown(
                        f"""<h1 style='color:yellow;'>Genre : <span style='color:white;'>{targets[0][0]}</span></h1>""",
                        unsafe_allow_html=True)




class StreamlitApp:


    # @st.cache
    def __init__(self):
        self.audio_style_embeddings = pickle.load(open("audio_style_embeddings.pickle","rb"))
        self.audio = pickle.load(open("audio.pickle","rb"))

        print(len(self.audio_style_embeddings))
        print(len(self.audio))
        st.write()


    def search_by_style(self, reference_audio, max_results):
        v0 = self.audio_style_embeddings[reference_audio]
        distances = {}
        for k,v in self.audio_style_embeddings.items():
            d = sc.spatial.distance.cosine(v0, v)
            distances[k] = d

        sorted_neighbors = sorted(distances.items(), key=lambda x: x[0], reverse=False)
        print("sorted_neighbors ===",sorted_neighbors)
        # f, ax = plt.subplots(1, max_results, figsize=(16, 8))
        # for i, img in enumerate(sorted_neighbors[:max_results]):
        #     ax[i].imshow(images[img[0]])
        #     ax[i].set_axis_off()
        st.subheader("Similar Music you may want to Listen")
        
        for i, aud in enumerate(sorted_neighbors[:max_results]): #list of images
        
            
            st.write(aud[0])
            #st.audio(aud[0], format = 'audio/wav')
        self.audio = pickle.load(open("audio.pickle","rb"))

        



    def asearch(self):
#        st.title("Music Recommendation System")
        st.write("-------------------------------------------------------------------------------------------------")

        
        st.subheader("UPLOAD AN AUDIO FILE: ")
        audio_file = st.file_uploader("Upload Audio", type=["wav"])
        k = st.slider('How many similar audio do you want to listen?', 1, 5, 1)
        

        st.write("Number of similar audio files selected ===>"+str(k))
        if audio_file is not None:

            # To See details
            file_details = {"filename":audio_file.name, "filetype":audio_file.type,
                          "filesize":audio_file.size}
            st.write(file_details)
            #aud = self.load_audio(audio_file) #line 21
            aud = st.audio(audio_file, format = 'audio/wav')
            # To View Uploaded Image
            #st.audio(audio_file, format="audio/wav", start_time=0)

            self.search_by_style(audio_file.name, k) #line27
    


if __name__ == "__main__":

    
    obj  = StreamlitApp()
    # obj.load_embeddings()
    obj.asearch()

