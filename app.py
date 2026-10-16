import ast
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


def get_names(text):
    if isinstance(text, str):
        text = ast.literal_eval(text)
    return [item["name"] for item in text]


@st.cache_resource
def load():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    for col in ["genres", "keywords", "production_companies"]:
        movies[col] = movies[col].apply(get_names)
    movies["tags"] = (movies["genres"] + movies["keywords"]).apply(
        lambda x: " ".join(x).lower()
    )
    vectors = CountVectorizer(max_features=5000, stop_words="english").fit_transform(
        movies["tags"]
    )
    return movies, vectors


movies, vectors = load()
studio_counts = Counter(s for studios in movies["production_companies"] for s in studios)

st.title("🎬 Movie Recommender")
title = st.selectbox("Pick a movie you like", movies["title"].sort_values())
st.write("You picked:", title)
def recommend(title, different_studio=False, n=5):
    i = movies[movies["title"] == title].index[0]
    scores = cosine_similarity(vectors[i], vectors).flatten()

    studios = movies.loc[i, "production_companies"]
    main_studio = min(studios, key=lambda s: studio_counts[s]) if studios else None

    picked = []
    for idx in scores.argsort()[::-1]:
        if idx == i:
            continue
        if different_studio and main_studio in movies.loc[idx, "production_companies"]:
            continue
        picked.append(idx)
        if len(picked) == n:
            break
    return movies.loc[picked]


different = st.toggle("Different studio (skip movies from the same main studio)")

if st.button("Recommend"):
    for _, row in recommend(title, different).iterrows():
        st.subheader(row["title"])
        st.write(row["overview"])