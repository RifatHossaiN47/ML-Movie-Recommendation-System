import pickle
import streamlit as st
import pandas as pd


movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
df = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))
def recommend(movie):
    movie_index = df[df['title']== movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x:x[1])[0:6]
    recommended_movie = []
    for i in movie_list:
        recommended_movie.append(df.iloc[i[0]].title)
    return recommended_movie



st.title('Movie Recommendation System')

selected_movies = st.selectbox('Select Movies', df['title'].values)

if st.button('Recommend'):
    recommendations = recommend(selected_movies)
    for i in recommendations:
        st.write(i)