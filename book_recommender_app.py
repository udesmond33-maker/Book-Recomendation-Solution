"""
Book Recommender — Redesigned to match Trivago app style
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
    page_title="Book Maestro",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #FFFFFF !important;
    color: #111111 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }
body { background-color: #FFFFFF !important; }
.stApp { background-color: #FFFFFF !important; }
section[data-testid="stSidebar"] { display: none; }

/* ── Header ── */
.br-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 0 1rem 0;
    border-bottom: 2px solid #1A3A5C;
    margin-bottom: 2rem;
}
.br-logo {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -1px;
    color: #1A3A5C;
    font-family: 'Inter', sans-serif;
}
.br-logo span { color: #C9923A; }
.br-tagline {
    font-size: 0.82rem;
    color: #666666;
    margin-top: 2px;
    font-weight: 400;
}
.br-badge {
    background: #1A3A5C;
    color: #FFFFFF;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.4rem 1rem;
    border-radius: 6px;
}

/* ── Metric cards ── */
.metric-card {
    background: #FAFAFA;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #E5E5E5;
    border-left: 4px solid #C9923A;
    margin-bottom: 1rem;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #1A3A5C;
    margin: 0;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.72rem;
    color: #777777;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 0.3rem 0 0 0;
    font-weight: 600;
}

/* ── Section titles ── */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #111111;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #F0F0F0;
}

/* ── Insight box ── */
.insight-box {
    background: #F5F7FF;
    border-left: 4px solid #1A3A5C;
    border-radius: 0 6px 6px 0;
    padding: 0.9rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.88rem;
    color: #111111;
    line-height: 1.65;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #F5F5F5;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid #E5E5E5;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 0.45rem 1.2rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: #444444 !important;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #1A3A5C !important;
    color: #FFFFFF !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #1A3A5C !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.2rem !important;
    font-weight: 500 !important;
    font-size: 0.87rem !important;
    width: 100% !important;
}
.stButton > button:hover { background: #0f2540 !important; }

/* ── Selected book banner ── */
.selected-banner {
    background: #F0F4FA;
    border: 1px solid #D0D9E8;
    border-left: 4px solid #C9923A;
    border-radius: 0 8px 8px 0;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.5rem;
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
    font-size: 1.15rem;
    font-weight: 700;
    color: #1A3A5C;
    margin: 0 0 2px 0;
}
.selected-author {
    font-size: 0.85rem;
    color: #636366;
    margin: 0 0 6px 0;
}
.selected-meta { font-size: 0.78rem; color: #AEAEB2; }

/* ── Book cards ── */
.book-card {
    background: #FAFAFA;
    border-radius: 8px;
    padding: 1rem 1.3rem;
    border: 1px solid #E5E5E5;
    border-left: 4px solid #1A3A5C;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}
.card-left { flex: 1; }
.card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #111111;
    margin: 0 0 3px 0;
}
.card-author {
    font-size: 0.82rem;
    color: #636366;
    margin: 0 0 4px 0;
}
.card-meta { font-size: 0.76rem; color: #AEAEB2; margin: 0; }
.card-right {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
    margin-left: 12px;
    flex-shrink: 0;
}
.match-pill {
    background: #1A3A5C;
    color: #FFFFFF;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 20px;
    white-space: nowrap;
}
.genre-pill {
    background: #FFF3E0;
    color: #C9923A;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 20px;
    white-space: nowrap;
}

/* ── Text input ── */
.stTextInput input {
    border: 1.5px solid #E5E5EA !important;
    border-radius: 6px !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
    color: #111111 !important;
}
.stTextInput input:focus {
    border-color: #1A3A5C !important;
    box-shadow: 0 0 0 3px rgba(26,58,92,0.1) !important;
}

/* ── Dropdowns — dark background, white text ── */
[data-baseweb="select"] div {
    background-color: #1A3A5C !important;
    color: #FFFFFF !important;
    border-radius: 6px !important;
    font-size: 0.9rem !important;
}
[data-baseweb="select"] span { color: #FFFFFF !important; }
[data-baseweb="select"] svg  { fill: #FFFFFF !important; }
[data-baseweb="popover"] li {
    background-color: #1A3A5C !important;
    color: #FFFFFF !important;
}
[data-baseweb="popover"] li:hover {
    background-color: #C9923A !important;
    color: #FFFFFF !important;
}
[data-baseweb="menu"] { background-color: #1A3A5C !important; }
[data-baseweb="menu"] [role="option"] {
    color: #FFFFFF !important;
    background-color: #1A3A5C !important;
}
[data-baseweb="menu"] [role="option"]:hover {
    background-color: #C9923A !important;
}

/* Force all text visible */
p, h1, h2, h3, h4, h5, label, span, div { color: #111111; }
.stMarkdown p { color: #111111 !important; }
.stSelectbox label, .stSlider label, .stRadio label { color: #111111 !important; }
[data-testid="stMarkdownContainer"] p { color: #111111 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="br-header">
  <div>
    <div class="br-logo">Book <span>Maestro</span></div>
    <div class="br-tagline">AI-powered book recommendations · TF-IDF + K-Means Clustering</div>
  </div>
  <div class="br-badge">Desmond Ugboaja &nbsp;·&nbsp; Ironhack Analytics 2026</div>
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
        st.error(f"Model file not found: {e}")
        st.stop()

@st.cache_data
def load_data():
    try:
        return pd.read_csv('clustered_books.csv')
    except FileNotFoundError:
        st.error("clustered_books.csv not found.")
        st.stop()

with st.spinner("Loading models…"):
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
      <div class="card-left">
        <p class="card-title">{row['title']}</p>
        <p class="card-author">by {row['authors']}</p>
        <p class="card-meta">{rating}Cluster {int(row['cluster'])}</p>
      </div>
      <div class="card-right">
        {match_html}
        {genre_html}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────────────────────
exclude = {'mixed', 'default', 'add a comment', 'unknown'}
genre_counts = df[~df['genre'].str.lower().isin(exclude)]['genre'].value_counts()

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, f"{len(df):,}",                  "Books in dataset"),
    (c2, f"{df['authors'].nunique():,}",  "Unique authors"),
    (c3, f"{df['cluster'].nunique()}",    "Content clusters"),
    (c4, f"{len(genre_counts)}",          "Genres covered"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <p class="metric-value">{val}</p>
      <p class="metric-label">{lbl}</p>
    </div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  🔍  Search by Title  ", "  📖  Browse by Genre  "])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — SEARCH BY TITLE
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Find books similar to one you love</p>',
                unsafe_allow_html=True)

    col_input, col_n, col_btn = st.columns([3, 1.5, 1])
    with col_input:
        query = st.text_input(
            "title",
            placeholder="Type a title — e.g. Born a Crime, Sharp Objects, The Alchemist…",
            label_visibility="collapsed"
        )
    with col_n:
        n_recs = st.slider("Results", 3, 10, 5, key="s1")
    with col_btn:
        st.markdown("<div style='margin-top:1.85rem'></div>", unsafe_allow_html=True)
        search = st.button("Search", key="btn_search")

    if search and query:
        matches = df[df['title'].str.contains(query, case=False, na=False)]

        if matches.empty:
            st.warning(f"No titles found matching **'{query}'**. Try a single keyword.")
        else:
            if len(matches) > 1:
                chosen   = st.selectbox(f"{len(matches)} matches — pick one:", matches['title'].tolist())
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

            st.markdown('<p class="section-title">You might also enjoy</p>', unsafe_allow_html=True)
            recs = get_recommendations(book_idx, n=n_recs)
            for _, row in recs.iterrows():
                render_card(row, show_match=True)

            top          = recs.iloc[0]
            cluster_size = (df['cluster'] == book['cluster']).sum()
            st.markdown(f"""
            <div class="insight-box">
            <b>Why these books?</b> &nbsp;<b>{book['title']}</b> belongs to Cluster {int(book['cluster'])}
            — a group of <b>{cluster_size:,} books</b> with similar themes and vocabulary.
            Recommendations are ranked by cosine similarity within that cluster.
            Top match: <b>{top['title']}</b> at {top['match_pct']}% similarity.
            </div>
            """, unsafe_allow_html=True)

    elif search and not query:
        st.info("Enter a title above and hit Search.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — BROWSE BY GENRE
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Browse top-rated books by genre</p>',
                unsafe_allow_html=True)

    genre_list = genre_counts.index.tolist()

    col_g, col_s, col_n2 = st.columns([2.5, 1.5, 1])
    with col_g:
        selected_genre = st.selectbox(
            "Genre",
            genre_list,
            format_func=lambda x: f"{x}  ({genre_counts[x]:,} books)"
        )
    with col_s:
        sort_by = st.selectbox("Sort by", ["Highest rated", "Most rated", "Title A–Z"])
    with col_n2:
        n_browse = st.slider("Show", 5, 20, 10, key="s2")

    if selected_genre:
        gdf = df[df['genre'] == selected_genre].copy()
        if sort_by == "Highest rated":
            gdf = gdf[gdf['avg_rating'].notna()].nlargest(n_browse, 'avg_rating')
        elif sort_by == "Most rated":
            gdf = gdf[gdf['ratings_count'].notna()].nlargest(n_browse, 'ratings_count')
        else:
            gdf = gdf.sort_values('title').head(n_browse)

        st.markdown(f'<p class="section-title">Top {len(gdf)} books in {selected_genre}</p>',
                    unsafe_allow_html=True)

        if gdf.empty:
            st.info("No rated books found. Try sorting by Title A–Z.")
        else:
            for _, row in gdf.iterrows():
                render_card(row, show_match=False)

            st.markdown(f"""
            <div class="insight-box">
            <b>{selected_genre}</b> contains <b>{genre_counts.get(selected_genre, 0):,} books</b>
            in this dataset. Switch to the <b>Search by Title</b> tab and enter any title
            above to get content-based recommendations from within its cluster.
            </div>
            """, unsafe_allow_html=True)

st.caption("Book Recommender · Desmond Ugboaja · Ironhack Data Analytics 2026 · TF-IDF + K-Means + Cosine Similarity")
