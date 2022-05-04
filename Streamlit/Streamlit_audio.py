import streamlit as st
import IPython.display as ipd
#import cosine_similarity
from sklearn import preprocessing


st.header("Music Recommendation System")
st.write("Upload any music and get similar music")

uploaded_file = st.file_uploader("Choose an audio file...")

class StreamLitApp:

def asearch(self):
	st.title("SIMILAR MUSIC")
		st.write("-------------------------------------------------------------------------------------------------")

		st.subheader("UPLOAD MUSIC FILE: ")
		audio_file = st.file_uploader("Upload Music", type=["wav","mp3"])
		k = st.slider('How many similar songs do you want to listen?', 1, 6, 1)

		st.write("K value selected ===>"+str(k))
		if audio_file is not None:

			# To See details
			file_details = {"filename":audio_file.name, "filetype":audio_file.type,
						  "filesize":audio_file.size}
			st.write(file_details)
			img = self.load_image(audio_file)
			# To View Uploaded Image
			st.image(img,width=250)
			self.search_by_style(audio_file.name, k)



if __name__ == "__main__":
	
	obj  = StreamlitApp()
	# obj.load_embeddings()
	obj.asearch()