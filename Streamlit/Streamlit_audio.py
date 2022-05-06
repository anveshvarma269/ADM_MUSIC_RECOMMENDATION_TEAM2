import streamlit as st
import IPython.display as ipd
import pandas as pd
import numpy as np
import scipy as sc
import pickle
#from audio_feature.audio_featurizer import audio_process, spectrogram_plot
#from models.load_model import model_loader
#import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import manifold
from sklearn import preprocessing
from pydub import AudioSegment
import os
from scipy.io import wavfile



class StreamlitApp:


	# @st.cache
	def __init__(self):
		self.audio_style_embeddings = pickle.load(open("audio_style_embeddings.pickle","rb"))
		self.audio = pickle.load(open("audio.pickle","rb"))

		print(len(self.audio_style_embeddings))
		print(len(self.audio))

	@st.cache
	def load_audio(self,audio_file): #opening
		aud = st.open(audio_file)
		return aud


	def search_by_style(self, reference_audio, max_results):
		v0 = self.audio_style_embeddings[reference_audio]
		distances = {}
		for k,v in self.audio_style_embeddings.items():
			d = sc.spatial.distance.cosine(v0, v)
			distances[k] = d

		sorted_neighbors = sorted(distances.items(), key=lambda x: x[1], reverse=False)
		print("sorted_neighbors ===",sorted_neighbors)
		# f, ax = plt.subplots(1, max_results, figsize=(16, 8))
		# for i, img in enumerate(sorted_neighbors[:max_results]):
		#     ax[i].imshow(images[img[0]])
		#     ax[i].set_axis_off()
		st.subheader("Similar Music you may want to Listen")
		for i, aud in enumerate(sorted_neighbors[:max_results]): #list of images
			# st.write(aud[0])
			st.audio(aud[0], format="audio/wav", start_time=0)
			


			# st.audio(self.audio[aud[0]], width=100)#displaying all images
	    



	def asearch(self):
		st.title("Music Recommendation System")
		st.write("-------------------------------------------------------------------------------------------------")

		
		st.subheader("UPLOAD AN AUDIO FILE: ")
		audio_file = st.file_uploader("Upload Audio", type=["wav"])
		k = st.slider('How many similar audio do you want to listen?', 1, 6, 1)
		

		st.write("K value selected ===>"+str(k))
		if audio_file is not None:

			# To See details
			file_details = {"filename":audio_file.name, "filetype":audio_file.type,
						  "filesize":audio_file.size}
			st.write(file_details)
			#aud = self.load_audio(audio_file) #line 21
			aud = st.audio(audio_file, format = 'audio/wav')
			# To View Uploaded Image
			st.audio(audio_file, format="audio/wav", start_time=0)

			self.search_by_style(audio_file.name, k) #line27
	


if __name__ == "__main__":

	
	obj  = StreamlitApp()
	# obj.load_embeddings()
	obj.asearch()