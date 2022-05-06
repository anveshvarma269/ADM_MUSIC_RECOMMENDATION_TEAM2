import streamlit as st
import IPython.display as ipd
import pandas as pd
import numpy as np
#import scipy as sc
#from audio_feature.audio_featurizer import audio_process, spectrogram_plot
#from models.load_model import model_loader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import manifold
#from sklearn import preprocessing
#from pydub import AudioSegment
import os





class StreamlitApp:


	sim_df_names = pd.read_csv("update.csv", "rb")


	@st.cache
	def load_audio(self,audio_file):
		aud = Audio.open(image_file)
		return aud

	def find_similar_songs(name):
    # Find songs most similar to another song
	    series = sim_df_names[name].sort_values(ascending = False)
	    
	    # If we remove cosine similarity == 1 (songs will always have the best match with themselves)
	    series = series.drop(name)
	    
	    # Display the 5 top matches 
	    print("\n*******\nSimilar songs to ", name)
	    print(series.head(5))


	def asearch(self):
		st.header("Music Recommendation System")
		st.write("-------------------------------------------------------------------------------------------------")
		st.image(
		"https://q5n8c8q9.rocketcdn.me/wp-content/uploads/2018/08/The-20-Best-Royalty-Free-Music-Sites-in-2021.png.webp",
		width=800 
		)
		st.write("Upload any music and get similar music:")


		st.subheader("UPLOAD MUSIC FILE: ")
		audio_file = st.file_uploader("Upload Music", type=["wav","mp3"])
		k = st.slider('How many similar songs do you want to listen?', 1, 6,)
		

		st.write("Songs list ===>"+str(k))
		if audio_file is not None:

			# To See details
			file_details = {"filename":audio_file.name, "filetype":audio_file.type,
						  "filesize":audio_file.size, "audiofile":audio_file}
			st.write(file_details)
			aud = self.load_audio(audio_file)
			# To View Uploaded audio
			#st.audio(aud,width=250)
			self.find_similar_songs(audio_file.name, k)

if __name__ == "__main__":
	
	obj  = StreamlitApp()
	# obj.load_embeddings()
	obj.asearch()