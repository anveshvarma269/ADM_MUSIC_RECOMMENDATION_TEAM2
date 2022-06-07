# ADM_MUSIC_RECOMMENDATION_TEAM2


![image](https://user-images.githubusercontent.com/91440961/167236090-9bccae55-a072-4c41-925b-c24656022fc8.png)


**OVERVIEW**
 
The digitization of music, as well as the emergence of online streaming platforms and mobile apps, has transformed how we consume music. Much of the music  we listen to nowadays is based on our personal preferences. Each person has their own musical taste and preferred genre based on a variety of factors. This sparked the idea for a music recommendation system to assist people in discovering music they enjoy based on what they've already heard.

**WEB APPLICATION - HOSTED ON STREAMLIT**

[[https://share.streamlit.io/anveshvarma269/adm_music_recommendation_team2/main/app.py](https://share.streamlit.io/anveshvarma269/music-recommendation-and-genre-classification/main/app.py)](https://share.streamlit.io/anveshvarma269/music-recommendation-and-genre-classification/main/app.py)

 
**GOALS**
 
·   	To offer a simple web application to let users get recommendations to the desired music genres with the help of Streamlit Cloud


·   	To Classify music into multiple genres and give recommendations to the user based on the genre and type of music inputted


**Environment Setup**

Anaconda Installation

Install Anaconda:
Please go to the Anaconda.com/downloads site.
Select the respective platform: Windows/Mac/Linux.
Download the .exe installer.
Open and execute the .exe installer.
Launch Anaconda Navigator.
Click on the Install Jupyter Notebook Button.
Beginning the Installation.
Upon installation, open jupyter notebook.

Python Libraries

Open Anaconda console and install the following libraries

pip install scikit-learn
pip install tensorflow
pip install keras
conda install -c conda-forge xgboost
pip install librosa
pip install pydub 
pip install seaborn
 
**MAIN REQUIREMENTS**


·   	Python 3.8


·   	Pandas


·   	Numpy


·   	Streamlit


·   	Plotly


·   	Librosa


·   	Audio Featurizer 


·   	Cosine similarity


·   	Wavefile

 
**Use Cases**

 To give personalized music recommendations to users:

·Can predict and offer the appropriate songs to their users based on the characteristics of the music that has been heard previously.


·Can Address the Genre of music before recommending
 
**Data**
 
To build a model for the mentioned statement we will use “GTZAN Dataset - Music Genre Classification” Dataset.


Dataset Source - GTZAN Dataset - Music Genre Classification | Kaggle


The dataset consists of a collection of 10 genres with 100 audio files each, all having a length of 30 seconds and containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models).


**Process Outline**


·Feature Extraction: Chroma frequencies, Zero-crossing rates,Spectral - Centroid to classify music into multiple genres


·Model Building : 1. XGBoost technique 2. Convolutional Neural Networks. Both the classifiers gave noteworthy results with great accuracy and least margin of error. Which suggests it can be used on real world data as well.


·Recommender Systems enable us for any given vector to find the best similarity, ranked in descending order, from the best match to the least best match   and for audio files this will be done by cosine similarity


·Designed a pipeline and system to implement this approach and discussion on the system’s capabilities


·Deploy the application using stream lit to demonstrate the prediction and recommendation results

