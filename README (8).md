# 📚 Book Maestro

**AI-powered book recommendation system built with NLP and unsupervised machine learning.**

> TF-IDF · K-Means Clustering · Cosine Similarity · Streamlit  
> Desmond Ugboaja · Ironhack Data Analytics Bootcamp 2026

---

## What it does

Book Maestro recommends books based on content similarity — no user history required. Enter a title you love and the system returns the most similar books ranked by match percentage. You can also browse top-rated books by genre.

The recommendation engine works by:
1. Converting each book's text metadata into a numerical vector using TF-IDF
2. Grouping similar books into clusters using K-Means
3. Ranking books within the same cluster by cosine similarity to your chosen title

---

## Live demo

🚀 **[book-maestro.streamlit.app](https://book-maestro.streamlit.app)** ← replace with your actual URL after deploying

---

## Project structure

```
book-maestro/
│
├── book_recommender_app.py   # Streamlit app — main entry point
├── clustered_books.csv       # Dataset: 2,591 books with clusters and metadata
├── tfidf_model.pkl           # Fitted TF-IDF vectoriser
├── kmeans_model.pkl          # Fitted K-Means model
├── tfidf_matrix.npz          # Sparse TF-IDF matrix (scipy format)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## How to run locally

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/book-maestro.git
cd book-maestro
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run book_recommender_app.py
```

The app opens at `http://localhost:8501`

---

## Dataset

2,591 unique books collected from three sources:

| Source | Method | Books | Notes |
|--------|--------|-------|-------|
| books.toscrape.com | Web scraping (requests + BeautifulSoup) | 999 | Title, genre, price |
| Open Library API | REST API (30 subjects × 100 books) | ~1,400 | Ratings, authors, subjects |
| Gutendex API | REST API (Project Gutenberg classics) | ~200 | No API key required |

After deduplication: **2,591 unique books** across 58 genres.

---

## How the model works

### 1. Feature engineering
A `content` field was created by concatenating: `title + authors + genre + subjects`. This treats each book as a single text document for NLP processing.

### 2. TF-IDF vectorisation
- Library: `sklearn.feature_extraction.text.TfidfVectorizer`
- Output: sparse matrix of shape `(2591, 5000)`
- Words distinctive to specific books score high; words common across all books score low

### 3. K-Means clustering
- Library: `sklearn.cluster.KMeans`
- Optimal K selected using the Elbow Method
- Books in the same cluster share similar themes, genres, and vocabulary

### 4. Recommendation
- Input book → identify its cluster → compute cosine similarity against all books in that cluster → return top N ranked results with match percentage

### Why cosine similarity and not Euclidean distance?
Cosine similarity measures the *angle* between two vectors, not their magnitude. For TF-IDF text data, the direction of a vector (which words appear) matters more than its length (how many words). A short book and a long book on the same topic should be considered similar — cosine handles this correctly.

---

## App features

**Search by Title tab**
- Type any book title (partial match supported)
- Adjust number of recommendations (3–10)
- See match percentage, genre, and cluster for each result
- Insight box explains why those books were recommended

**Browse by Genre tab**
- Select from 58 genres
- Sort by highest rated, most rated, or alphabetical
- Configurable number of results (5–20)

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Web scraping | requests, BeautifulSoup4 |
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn |
| Sparse matrices | scipy |
| App framework | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Requirements

```
streamlit
pandas
numpy
scikit-learn
scipy
```

Install with: `pip install -r requirements.txt`

---

## Limitations and future improvements

**Current limitations**
- Scraped books (999) lack author data — genre used as fallback for NLP content field
- Content-based only — no collaborative filtering (user behaviour not used)
- Book descriptions not included — only title, author, genre, and subjects

**Planned improvements**
- Add collaborative filtering layer using user rating data
- Integrate full book descriptions via Google Books API for richer TF-IDF features
- Add book cover images via Open Library Covers API
- User session history to personalise recommendations over time

---

## Author

**Desmond Ugboaja**  
Senior Business Analyst · Data Analytics  
[linkedin.com/in/desmond-ugboaja-4346911b](https://linkedin.com/in/desmond-ugboaja-4346911b)  
Ironhack Data Analytics Bootcamp · June 2026

---

## Licence

MIT — free to use, modify, and distribute with attribution.
