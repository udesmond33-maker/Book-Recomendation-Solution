"""
Book Recommender — Redesigned UI
Clean, minimal, professional
Desmond Ugboaja · Ironhack Data Analytics 2026
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
    layout="centered"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #FFFFFF;
    color: #1C1C1E;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 3rem 2rem 4rem 2rem;
    max-width: 720px;
}

/* ── Header ── */
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #1C1C1E;
    margin: 0 0 4px 0;
    letter-spacing: -0.3px;
}
.page-sub {
    font-size: 0.85rem;
    color: #8E8E93;
    margin: 0 0 2.5rem 0;
    font-weight: 400;
}
.divider {
    border: none;
    border-top: 1px solid #E5E5EA;
    margin: 2rem 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #E5E5EA;
    background: transparent;
    padding: 0;
    margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    padding: 0.6rem 1.5rem 0.6rem 0;
    margin-bottom: -2px;
    font-size: 0.9rem;
    font-weight: 500;
    color: #8E8E93;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    border-bottom: 2px solid #C9923A !important;
    color: #1C1C1E !important;
}

/* ── Search input ── */
.stTextInput input {
    border: 1.5px solid #E5E5EA !important;
    border-radius: 10px !important;
    padding: 0.7rem 1rem !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
    background: #FAFAFA !important;
    color: #1C1C1E !important;
    box-shadow: none !important;
}
.stTextInput input:focus {
    border-color: #C9923A !important;
    background: #FFFFFF !important;
    box-shadow: 0 0 0 3px rgba(201,146,58,0.12) !important;
}

/* ── Button ── */
.stButton > button {
    background: #1C1C1E !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.8rem !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.2px !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* ── Selected book banner ── */
.selected-banner {
    background: #FAF8F5;
    border: 1px solid #E5E5EA;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.8rem;
}
.selected-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #C9923A;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin: 0 0 6px 0;
}
.selected-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    color: #1C1C1E;
    margin: 0 0 2px 0;
}
.selected-author {
    font-size: 0.85rem;
    color: #636366;
    margin: 0 0 6px 0;
}
.selected-meta {
    font-size: 0.78rem;
    color: #AEAEB2;
}

/* ── Book card ── */
.book-card {
    background: #FAF8F5;
    border: 1px solid #E5E5EA;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.15s;
}
.book-card:hover { border-color: #C9923A; }
.card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}
.card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1C1C1E;
    margin: 0 0 3px 0;
    line-height: 1.35;
}
.card-author {
    font-size: 0.82rem;
    color: #636366;
    margin: 0;
}
.card-meta {
    font-size: 0.76rem;
    color: #AEAEB2;
    margin: 6px 0 0 0;
}
.match-pill {
    background: #1C1C1E;
    color: #FFFFFF;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 20px;
    white-space: nowrap;
    margin-left: 10px;
    flex-shrink: 0;
}
.genre-pill {
    background: #FFF3E0;
    color: #C9923A;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 20px;
    margin-left: 6px;
    flex-shrink: 0;
}

/* ── Insight box ── */
.insight {
    background: #FFF8F0;
    border: 1px solid #FDDCB0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.83rem;
    color: #636366;
    margin-top: 1.5rem;
    line-height: 1.6;
}
.insight b { color: #1C1C1E; }

/* ── Section heading ── */
.section-heading {
    font-size: 1rem;
    font-weight: 600;
    color: #1C1C1E;
    margin: 0 0 1rem 0;
}

/* ── Selectbox ── */
.stSelectbox [data-baseweb="select"] > div {
    border: 1.5px solid #E5E5EA !important;
    border-radius: 10px !important;
    background: #FAFAFA !important;
    font-size: 0.9rem !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { padding: 0.5rem 0; }

/* ── Stats ── */
.stats-row {
    display: flex;
    gap: 2rem;
    margin-top: 1rem;
}
.stat-item { text-align: center; }
.stat-value {
    font-size: 1.4rem;
    font-weight: 600;
    color: #1C1C1E;
    display: block;
}
.stat-label {
    font-size: 0.75rem;
    color: #AEAEB2;
    font-weight: 400;
}
.footer-text {
    font-size: 0.75rem;
    color: #C7C7CC;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<p class="page-title">📚 Book Recommender</p>
<p class="page-sub">Find your next read · TF-IDF + K-Means · Ironhack 2026</p>
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
        st.error(f"Model file not found: {e}")
        st.stop()

@st.cache_data
def load_data():
    try:
        return pd.read_csv('clustered_books.csv')
    except FileNotFoundError:
        st.error("clustered_books.csv not found.")
        st.stop()

with st.spinner("Loading…"):
    tfidf, kmeans, tfidf_matrix = load_models()
    df = load_data()

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_recommendations(book_idx, n=5):
    cluster  = df.loc[book_idx, 'cluster']
    pool_idx = df[df['cluster'] == cluster].index.tolist()
    book_vec = tfidf_matrix[book_idx]
    pool_mat = tfidf_matrix[pool_idx]
    sims     = cosine_similarity(book_vec, pool_mat).flatten()
    sim_s    = pd.Series(sims, index=pool_idx).drop(book_idx, errors='ignore')
    top_idx  = sim_s.nlargest(n).index
    results  = df.loc[top_idx].copy()
    results['similarity'] = sim_s[top_idx].values
    results['match_pct']  = (results['similarity'] * 100).round(1)
    return results

def render_card(row, show_match=True):
    rating = f"⭐ {row['avg_rating']:.2f}  ·  " if pd.notna(row.get('avg_rating')) else ""
    match_html = f'<span class="match-pill">{row["match_pct"]}% match</span>' if show_match and 'match_pct' in row else ''
    genre_html = f'<span class="genre-pill">{row["genre"]}</span>'
    st.markdown(f"""
    <div class="book-card">
      <div class="card-top">
        <p class="card-title">{row['title']}</p>
        <div style="display:flex;align-items:center;margin-top:2px">{match_html}{genre_html}</div>
      </div>
      <p class="card-author">by {row['authors']}</p>
      <p class="card-meta">{rating}Cluster {int(row['cluster'])}</p>
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Search by title", "Browse by genre"])

# ── TAB 1: Search ─────────────────────────────────────────────────────────────
with tab1:
    query = st.text_input(
        "title_search",
        placeholder="Type a book title — e.g. Born a Crime, Sharp Objects…",
        label_visibility="collapsed"
    )
    col_n, col_btn = st.columns([3, 1])
    with col_n:
        n_recs = st.slider("Recommendations", 3, 10, 5, key="s1")
    with col_btn:
        st.write("")
        search = st.button("Search", key="btn_search")

    if search and query:
        matches = df[df['title'].str.contains(query, case=False, na=False)]

        if matches.empty:
            st.warning(f"No titles found matching **'{query}'**. Try a single keyword.")
        else:
            if len(matches) > 1:
                chosen = st.selectbox(
                    f"{len(matches)} matches — pick one:",
                    matches['title'].tolist()
                )
                book     = df[df['title'] == chosen].iloc[0]
                book_idx = df[df['title'] == chosen].index[0]
            else:
                book     = matches.iloc[0]
                book_idx = matches.index[0]

            rating_txt = f"⭐ {book['avg_rating']:.2f}  ·  " if pd.notna(book.get('avg_rating')) else ""
            st.markdown(f"""
            <div class="selected-banner">
              <p class="selected-label">Searching from</p>
              <p class="selected-title">{book['title']}</p>
              <p class="selected-author">by {book['authors']}</p>
              <p class="selected-meta">{rating_txt}Genre: {book['genre']}  ·  Cluster {int(book['cluster'])}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<p class="section-heading">You might also enjoy</p>', unsafe_allow_html=True)
            recs = get_recommendations(book_idx, n=n_recs)
            for _, row in recs.iterrows():
                render_card(row, show_match=True)

            top = recs.iloc[0]
            cluster_size = (df['cluster'] == book['cluster']).sum()
            st.markdown(f"""
            <div class="insight">
            <b>{book['title']}</b> belongs to Cluster {int(book['cluster'])} — a group of
            <b>{cluster_size:,} books</b> with similar themes and vocabulary.
            Recommendations are ranked by cosine similarity within that cluster.
            Top match: <b>{top['title']}</b> at {top['match_pct']}% similarity.
            </div>
            """, unsafe_allow_html=True)

# ── TAB 2: Browse ─────────────────────────────────────────────────────────────
with tab2:
    exclude = {'mixed', 'default', 'add a comment', 'unknown'}
    genre_counts = (
        df[~df['genre'].str.lower().isin(exclude)]['genre']
        .value_counts()
    )
    genre_list = genre_counts.index.tolist()

    col_g, col_s = st.columns([3, 2])
    with col_g:
        selected_genre = st.selectbox(
            "Genre",
            genre_list,
            format_func=lambda x: f"{x}  ({genre_counts[x]:,} books)"
        )
    with col_s:
        sort_by = st.selectbox("Sort by", ["Highest rated", "Most rated", "Title A–Z"])

    n_browse = st.slider("Books to show", 5, 20, 10, key="s2")

    if selected_genre:
        gdf = df[df['genre'] == selected_genre].copy()
        if sort_by == "Highest rated":
            gdf = gdf[gdf['avg_rating'].notna()].nlargest(n_browse, 'avg_rating')
        elif sort_by == "Most rated":
            gdf = gdf[gdf['ratings_count'].notna()].nlargest(n_browse, 'ratings_count')
        else:
            gdf = gdf.sort_values('title').head(n_browse)

        st.markdown(f'<p class="section-heading">Top {len(gdf)} in {selected_genre}</p>', unsafe_allow_html=True)

        if gdf.empty:
            st.info("No rated books found here. Try sorting by Title A–Z.")
        else:
            for _, row in gdf.iterrows():
                render_card(row, show_match=False)

            st.markdown(f"""
            <div class="insight">
            <b>{selected_genre}</b> contains <b>{genre_counts.get(selected_genre, 0):,} books</b>
            in this dataset. Switch to the Search tab and enter any title above to get
            content-based recommendations from within its cluster.
            </div>
            """, unsafe_allow_html=True)

# ── Footer stats ──────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Books",    f"{len(df):,}")
c2.metric("Authors",  f"{df['authors'].nunique():,}")
c3.metric("Clusters", f"{df['cluster'].nunique()}")
c4.metric("Genres",   f"{len(genre_list)}")
st.markdown('<p class="footer-text">Desmond Ugboaja · Ironhack Data Analytics 2026 · TF-IDF + K-Means + Cosine Similarity</p>', unsafe_allow_html=True)
