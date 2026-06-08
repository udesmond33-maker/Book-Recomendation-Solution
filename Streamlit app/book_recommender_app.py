"""
Book Recommender — Phase 4: Streamlit App
==========================================
Uses saved models from Phase 3:
  - clustered_books.csv
  - tfidf_model.pkl
  - kmeans_model.pkl
  - tfidf_matrix.npz
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #FFFFFF;
    color: #1A1A2E;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1100px; }

.app-header {
    border-bottom: 3px solid #2C3E50;
    padding-bottom: 1rem;
    margin-bottom: 2rem;
}
.app-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #2C3E50;
    margin: 0;
}
.app-sub {
    font-size: 0.9rem;
    color: #666;
    margin: 0.2rem 0 0 0;
}

.book-card {
    background: #F8F9FA;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid #2C3E50;
    margin-bottom: 0.8rem;
}
.book-title {
    font-size: 1rem;
    font-weight: 600;
    color: #2C3E50;
    margin: 0;
}
.book-author {
    font-size: 0.85rem;
    color: #666;
    margin: 0.2rem 0 0 0;
}
.book-meta {
    font-size: 0.8rem;
    color: #888;
    margin: 0.4rem 0 0 0;
}
.match-badge {
    display: inline-block;
    background: #2C3E50;
    color: white;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 10px;
    margin-left: 8px;
}
.cluster-badge {
    display: inline-block;
    background: #27AE60;
    color: white;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 10px;
    margin-left: 4px;
}
.selected-book {
    background: #EBF5FB;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid #2980B9;
    margin-bottom: 1.5rem;
}
.stButton > button {
    background: #2C3E50 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 2rem !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}
.stButton > button:hover {
    background: #1A252F !important;
}
.insight-box {
    background: #EBF5FB;
    border-left: 4px solid #2980B9;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <p class="app-title">Book Recommender</p>
  <p class="app-sub">Content-based recommendations using TF-IDF and K-Means clustering · Ironhack Data Analytics 2026</p>
</div>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        with open('tfidf_model.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        matrix = sp.load_npz('tfidf_matrix.npz')
        return tfidf, kmeans, matrix
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Make sure tfidf_model.pkl, kmeans_model.pkl, and tfidf_matrix.npz are in the same folder.")
        st.stop()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('clustered_books.csv')
        return df
    except FileNotFoundError:
        st.error("clustered_books.csv not found. Run the nlp_clustering.ipynb notebook first.")
        st.stop()

with st.spinner("Loading models..."):
    tfidf, kmeans, tfidf_matrix = load_models()
    df = load_data()

# ── How it works expander ─────────────────────────────────────────────────────
with st.expander("How does this work?"):
    st.markdown("""
    **Step 1 — TF-IDF Vectorisation**
    Each book's title, author, genre and subjects are converted into a numerical vector.
    Words that appear in many books (e.g. 'fiction') get low scores.
    Distinctive words (e.g. 'wizardry') get high scores.

    **Step 2 — K-Means Clustering**
    Books are grouped into clusters based on their TF-IDF vectors.
    Similar books end up in the same cluster.

    **Step 3 — Cosine Similarity**
    When you search for a book, the system finds all books in the same cluster
    and ranks them by cosine similarity — how closely their vectors point in the same direction.
    """)

# ── Search ────────────────────────────────────────────────────────────────────
st.markdown("### Find a book you love")

col_input, col_btn = st.columns([4, 1])
with col_input:
    query = st.text_input(
        "Type a book title",
        placeholder="e.g. Harry Potter, Pride and Prejudice, Dune...",
        label_visibility="collapsed"
    )
with col_btn:
    search = st.button("Get recommendations")

n_recs = st.slider("Number of recommendations", 3, 10, 5)

# ── Recommendation logic ──────────────────────────────────────────────────────
def get_recommendations(title_query, n=5):
    matches = df[df['title'].str.contains(title_query, case=False, na=False)]

    if matches.empty:
        return None, None

    book    = matches.iloc[0]
    book_idx = matches.index[0]
    cluster = book['cluster']

    # Get all books in same cluster
    pool_idx = df[df['cluster'] == cluster].index.tolist()

    # Compute cosine similarity within cluster
    book_vec     = tfidf_matrix[book_idx]
    pool_matrix  = tfidf_matrix[pool_idx]
    similarities = cosine_similarity(book_vec, pool_matrix).flatten()

    # Rank and exclude the input book
    sim_series  = pd.Series(similarities, index=pool_idx)
    sim_series  = sim_series.drop(book_idx, errors='ignore')
    top_indices = sim_series.nlargest(n).index

    results = df.loc[top_indices].copy()
    results['similarity'] = sim_series[top_indices].values
    results['match_pct']  = (results['similarity'] * 100).round(1)

    return book, results

# ── Display results ───────────────────────────────────────────────────────────
if search and query:
    selected, recs = get_recommendations(query, n=n_recs)

    if selected is None:
        st.warning(f"No book found matching **'{query}'**. Try a different title.")
    else:
        # Show selected book
        rating_txt = f"⭐ {selected['avg_rating']:.2f}" if pd.notna(selected.get('avg_rating')) else ""
        st.markdown(f"""
        <div class="selected-book">
          <p style="font-size:0.75rem;color:#666;margin:0 0 4px 0">YOU SEARCHED FOR</p>
          <p class="book-title" style="font-size:1.1rem">{selected['title']}</p>
          <p class="book-author">by {selected['authors']}</p>
          <p class="book-meta">
            Genre: {selected['genre']} &nbsp;·&nbsp;
            Cluster: {int(selected['cluster'])} &nbsp;·&nbsp;
            {rating_txt}
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"### {n_recs} books you might enjoy")

        for _, row in recs.iterrows():
            rating_str = f"⭐ {row['avg_rating']:.2f}" if pd.notna(row.get('avg_rating')) else "No rating"
            st.markdown(f"""
            <div class="book-card">
              <p class="book-title">
                {row['title']}
                <span class="match-badge">{row['match_pct']}% match</span>
                <span class="cluster-badge">Cluster {int(row['cluster'])}</span>
              </p>
              <p class="book-author">by {row['authors']}</p>
              <p class="book-meta">
                {rating_str} &nbsp;·&nbsp; Genre: {row['genre']}
              </p>
            </div>
            """, unsafe_allow_html=True)

        # Insight
        top = recs.iloc[0]
        st.markdown(f"""
        <div class="insight-box">
        <b>Why these books?</b> <b>{selected['title']}</b> belongs to
        Cluster {int(selected['cluster'])} — a group of {(df['cluster'] == selected['cluster']).sum():,}
        similar books. Recommendations are ranked by cosine similarity — how closely
        each book's content vector aligns with <b>{selected['title']}</b>.
        Top match: <b>{top['title']}</b> at {top['match_pct']}% similarity.
        </div>
        """, unsafe_allow_html=True)

# ── Dataset stats ─────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Books in database", f"{len(df):,}")
c2.metric("Unique authors",    f"{df['authors'].nunique():,}")
c3.metric("Clusters",          f"{df['cluster'].nunique()}")
c4.metric("Genres",            f"{df['genre'].nunique()}")

st.caption("Book Recommender · Desmond Ugboaja · Ironhack Data Analytics 2026 · TF-IDF + K-Means + Cosine Similarity")
