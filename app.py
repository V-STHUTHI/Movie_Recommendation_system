import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="MovieMind | Recommender", page_icon="🎬", layout="wide")

# Custom CSS for Professional UI and "Designed by Stuthi" Footer
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        border: none;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .movie-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 15px;
        border-bottom: 3px solid #ff4b4b;
        margin-bottom: 20px;
        min-height: 220px;
    }
    .footer {
        width: 100%;
        text-align: center;
        color: #666;
        padding: 20px;
        font-size: 0.9em;
        border-top: 1px solid #333;
        margin-top: 50px;
    }
    h4 { color: #ffffff !important; margin-bottom: 5px; }
    p { color: #bfc5d1 !important; font-size: 0.85em; margin-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING & CLEANING ---
@st.cache_resource
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_path = os.path.join(base_dir, "IMDB_Movies_Dataset.csv")
    if not os.path.exists(df_path):
        df_path = r"C:\Users\sthut\Downloads\IMDB_Movies_Dataset.csv"

    df = pd.read_csv(df_path)
    df.columns = [c.strip() for c in df.columns]
    
    # Mapping variations in CSV column names
    rename_map = {
        'Rating': ['Rating', 'Average Rating', 'Score', 'imdb_rating', '_5'],
        'Director': ['Director', 'Directors', 'Filmmaker'],
        'Title': ['Title', 'Movie_Title', 'Names'],
        'Cast': ['Cast', 'Actors', 'Stars']
    }
    
    for target, options in rename_map.items():
        for opt in options:
            if opt in df.columns:
                df.rename(columns={opt: target}, inplace=True)
                break

    for col in ['Director', 'Cast', 'Title', 'Rating']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Feature engineering for similarity
    df['features'] = (df['Director'].astype(str) + ' ' + 
                      df['Cast'].astype(str) + ' ' + 
                      df['Title'].astype(str)).str.lower()
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return df, cosine_sim

# --- 3. RECOMMENDATION ENGINE ---
def get_recommendations(title, cosine_sim_matrix, df_data, num_recs):
    if title not in df_data['Title'].values:
        return pd.DataFrame()
    idx = df_data[df_data['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:num_recs+1]]
    return df_data.iloc[movie_indices]

# --- 4. FRONTEND UI ---
try:
    df, cosine_sim_matrix = load_data()

    st.title("🎬 MovieMind Pro")
    st.write("Intelligent discovery engine for cinema enthusiasts.")

    # Sidebar
    st.sidebar.header("Navigation")
    num_recs = st.sidebar.slider("Number of suggestions:", 3, 15, 6)
    
    
    # Main Interaction
    movie_titles = df['Title'].sort_values().tolist()
    selected_movie = st.selectbox("Search for a movie you enjoyed:", movie_titles)

    if st.button("✨ Generate Recommendations"):
        recs = get_recommendations(selected_movie, cosine_sim_matrix, df, num_recs)
        
        if not recs.empty:
            st.write(f"### Top Matches for '{selected_movie}':")
            cols = st.columns(3)
            for i, row in enumerate(recs.itertuples()):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <h4>{row.Title}</h4>
                        <p><b>Director:</b> {row.Director}</p>
                        <p><b>Cast:</b> {str(row.Cast)[:75]}...</p>
                        <hr style="border:0.1px solid #444">
                        <span style="color:#f1c40f; font-weight:bold;">⭐ Rating: {row.Rating}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Movie not found in the dataset.")

except Exception as e:
    st.error(f"Error: {e}")

# --- 5. FOOTER ---
st.markdown("""
    <div class="footer">
        <p>Developed with ❤️ for Data Science Project</p>
        <p><b>Designed by Sthuthi</b></p>
    </div>
    """, unsafe_allow_html=True)