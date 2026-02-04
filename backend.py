import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# 1. Page Configuration
st.set_page_config(page_title="IMDB Movie Recommender", page_icon="🎬", layout="wide")

# 2. Optimized Data and Model Loading
@st.cache_resource
def load_data_and_model():
    # Automatically finds the CSV in the same folder as this app.py script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_path = os.path.join(base_dir, "IMDB_Movies_Dataset.csv")
    
    # Load dataset 
    df = pd.read_csv(df_path)

    # Handle missing values in critical columns 
    for col in ['Director', 'Cast', 'Languages', 'Title']:
        df[col] = df[col].fillna('')

    # Create combined 'features' column for similarity 
    # We combine these to find links between directors, actors, and languages
    df['features'] = (df['Director'] + ' ' + df['Cast'] + ' ' + 
                      df['Languages'] + ' ' + df['Title']).str.lower()
    df['features'] = df['features'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # TF-IDF Vectorization: Converts text into a mathematical matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['features'])

    # Compute Cosine Similarity matrix: Measures the "angle" between movie vectors
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return df, cosine_sim

# 3. Recommendation Function 
def get_recommendations(title, cosine_sim_matrix, df_data, num_recommendations=10):
    if title not in df_data['Title'].values:
        return pd.DataFrame()

    # Get the index of the selected movie 
    idx = df_data[df_data['Title'] == title].index[0]

    # Get pairwise similarity scores for all movies compared to this one
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))

    # Sort movies by similarity score (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top N similar movies (excluding the movie itself at index 0) 
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]

    return df_data.iloc[movie_indices]

# --- MAIN EXECUTION ---

try:
    # Load assets
    df, cosine_sim_matrix = load_data_and_model()

    # Streamlit UI Design 
    st.title("🎬 Movie Recommendation System")
    st.markdown("Discover movies similar to your favorites based on **Director**, **Cast**, and **Language**.")

    # Sidebar for settings
    st.sidebar.header("Recommendation Settings")
    num_recs = st.sidebar.slider("Number of movies to suggest:", 5, 20, 10)

    # Search and selection
    movie_titles = df['Title'].sort_values().tolist()
    selected_movie = st.selectbox("Select or type a movie you like:", movie_titles)

    if st.button("Generate Recommendations"):
        with st.spinner(f"Analyzing {len(df)} movies..."):
            recommendations = get_recommendations(selected_movie, cosine_sim_matrix, df, num_recs)
            
            if not recommendations.empty:
                st.subheader(f"Top {num_recs} Suggestions for '{selected_movie}':")
                
                # Display results in a clean list
                for i, row in enumerate(recommendations.itertuples(), 1):
                    with st.container():
                        st.write(f"**{i}. {row.Title}**")
                        # Adjust _5 if your rating column has a different header name
                        st.caption(f"Director: {row.Director} | Languages: {row.Languages}")
                        st.divider()
            else:
                st.warning("No recommendations found. Try a different movie!")

except FileNotFoundError:
    st.error("Error: 'IMDB_Movies_Dataset.csv' not found.")
    st.info(f"Please move the CSV from your Downloads folder to: {os.path.dirname(os.path.abspath(__file__))}")

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed for Computer Applications data science project.")