import sqlite3
import arxiv
import os
import time
import requests

DB_PATH = os.path.join("data", "papers.db")

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            summary TEXT,
            source TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS query_cache (
            query TEXT PRIMARY KEY,
            last_fetched INTEGER
        )
    ''')
    conn.commit()
    conn.close()


def fetch_arxiv(query, max_results=20):
    print(f"Fetching from Arxiv: '{query}' ...")
    start_time = time.time()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    results = []
    for result in search.results():
        paper_id = result.entry_id
        title = result.title.strip()
        authors = ", ".join([a.name for a in result.authors])
        summary = result.summary.strip()
        results.append((paper_id, title, authors, summary, "arxiv"))

    end_time = time.time()
    print(f"Found {len(results)} papers in {end_time - start_time:.2f} seconds.")
    return results


def fetch_semantic_scholar(query, max_results=20):
    print(f"Fetching from Semantic Scholar: '{query}' ...")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    query_params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,abstract,url"
    }

    try:
        response = requests.get(url, params=query_params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for paper in data.get('data', []):
            paper_id = paper.get('paperId', '')
            title = paper.get('title') or ''
            abstract = paper.get('abstract') or ''
            authors_list = paper.get('authors', [])
            authors = ", ".join(a.get('name', '') for a in authors_list)

            # Skip paper if title or abstract is missing
            if not title.strip() or not abstract.strip():
                print(f"Skipping paper with missing title/abstract: {paper_id}")
                continue

            results.append((paper_id, title.strip(), authors, abstract.strip(), "semantic scholar"))

        print(f"Found {len(results)} papers.")
    except Exception as e:
        print(f"Error fetching from Semantic Scholar: {e}")
        results = []

    return results


def fetch_openalex(query, max_results=20):
    print(f"Fetching from OpenAlex: '{query}' ...")
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": max_results,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            paper_id = item.get("id", "")
            title = item.get("title") or ""
            abstract = item.get("abstract_inverted_index")
            authorships = item.get("authorships", [])
            authors = ", ".join(a.get("author", {}).get("display_name", "") for a in authorships)

            # Convert abstract_inverted_index to string
            if isinstance(abstract, dict):
                # Inverted index: word -> positions
                words = [None] * (max(pos for positions in abstract.values() for pos in positions) + 1)
                for word, positions in abstract.items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(w for w in words if w)
            else:
                abstract = ""

            # Skip if no usable title or abstract
            if not title.strip() or not abstract.strip():
                print(f"Skipping OpenAlex paper with missing title/abstract: {paper_id}")
                continue

            results.append((paper_id, title.strip(), authors, abstract.strip(), "openalex"))

        print(f"Found {len(results)} papers.")
    except Exception as e:
        print(f"Error fetching from OpenAlex: {e}")
        results = []

    return results


def save_papers(papers):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for paper in papers:
        try:
            c.execute(
                "INSERT OR IGNORE INTO papers (id, title, authors, summary, source) VALUES (?, ?, ?, ?, ?)",
                paper
            )
        except Exception as e:
            print(f"Error saving paper {paper[0]}: {e}")
    conn.commit()
    conn.close()


def query_papers_from_db(query):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT * FROM papers WHERE title LIKE ? OR summary LIKE ?",
        (f'%{query}%', f'%{query}%')
    )
    rows = c.fetchall()
    conn.close()
    return rows


def get_all_papers_from_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM papers")
    rows = c.fetchall()
    conn.close()
    return rows


def get_query_last_fetched(query):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT last_fetched FROM query_cache WHERE query = ?", (query,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def update_query_timestamp(query):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO query_cache (query, last_fetched) VALUES (?, ?)",
        (query, int(time.time()))
    )
    conn.commit()
    conn.close()
